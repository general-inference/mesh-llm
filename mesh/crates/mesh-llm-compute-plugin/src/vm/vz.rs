//! Apple's Virtualization.framework, on Apple Silicon.
//!
//! The guest boots the same kernel and image as on Linux, with a virtio
//! console, two virtio block devices and a virtio socket device. Every call
//! on a `VZVirtualMachine` must happen on the serial dispatch queue it was
//! created with, so each VM owns one queue and all framework calls are sent
//! to it; results come back over channels.
//!
//! The process needs the `com.apple.security.virtualization` entitlement.
//! `just compute-sign` signs a development build with it.

use std::os::fd::{FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use block2::RcBlock;
use dispatch2::{DispatchQueue, DispatchRetained};
use mesh_llm_compute_guest::protocol::{AGENT_PORT, EGRESS_PORT};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{AnyThread, DefinedClass, Message, define_class, msg_send};
use objc2_foundation::{NSArray, NSError, NSObject, NSObjectProtocol, NSString, NSURL};
use objc2_virtualization::{
    VZDiskImageStorageDeviceAttachment, VZEntropyDeviceConfiguration, VZFileSerialPortAttachment,
    VZLinuxBootLoader, VZSerialPortConfiguration, VZSocketDeviceConfiguration,
    VZStorageDeviceConfiguration, VZVirtioBlockDeviceConfiguration,
    VZVirtioConsoleDeviceSerialPortConfiguration, VZVirtioEntropyDeviceConfiguration,
    VZVirtioSocketConnection, VZVirtioSocketDevice, VZVirtioSocketDeviceConfiguration,
    VZVirtioSocketListener, VZVirtioSocketListenerDelegate, VZVirtualMachine,
    VZVirtualMachineConfiguration,
};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot};

use super::{Backend, BoxStream, EgressHandler, LaunchSpec, Vm, kernel_cmdline};

const CONSOLE_LOG: &str = "console.log";
const START_TIMEOUT: Duration = Duration::from_secs(30);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const STOP_TIMEOUT: Duration = Duration::from_secs(10);

pub struct VzBackend;

impl VzBackend {
    pub fn detect() -> Result<Self> {
        if !cfg!(target_arch = "aarch64") {
            bail!("sandboxes on macOS need Apple Silicon");
        }
        // SAFETY: a class property with no preconditions.
        if !unsafe { VZVirtualMachine::isSupported() } {
            bail!("this Mac does not support virtualization");
        }
        Ok(Self)
    }
}

#[async_trait]
impl Backend for VzBackend {
    fn name(&self) -> &'static str {
        "vz"
    }

    async fn launch(&self, spec: LaunchSpec) -> Result<Box<dyn Vm>> {
        let scratch = super::create_scratch(&spec.work_dir, spec.scratch_bytes)?;
        let console = spec.work_dir.join(CONSOLE_LOG);
        std::fs::File::create(&console)?;
        let paths = VmPaths {
            kernel: spec.kernel.clone(),
            rootfs: spec.rootfs.clone(),
            scratch,
            console: console.clone(),
        };
        let queue = DispatchQueue::new(&format!("mesh-compute.{}", spec.id), None);
        let (started_tx, started_rx) = oneshot::channel();
        let (egress_tx, egress_rx) = mpsc::unbounded_channel();
        let cpus = spec.cpus;
        let memory = spec.memory_mib << 20;
        let machine: Arc<Mutex<Option<OnQueue<Machine>>>> = Arc::new(Mutex::new(None));
        let slot = machine.clone();
        let vm_queue = queue.clone();
        queue.exec_async(move || {
            let result = Machine::boot(&vm_queue, &paths, cpus, memory, egress_tx, started_tx);
            match result {
                Ok(booted) => {
                    if let Ok(mut slot) = slot.lock() {
                        *slot = Some(OnQueue(booted));
                    }
                }
                Err(error) => error.report(),
            }
        });
        let started = tokio::time::timeout(START_TIMEOUT, started_rx)
            .await
            .context("the VM did not start in time")?
            .context("the VM was dropped before it started")?;
        if let Err(message) = started {
            bail!("start the VM: {message}");
        }
        tokio::spawn(forward_egress(egress_rx, spec.on_egress.clone()));
        Ok(Box::new(VzVm {
            queue,
            machine,
            console,
        }))
    }
}

struct VmPaths {
    kernel: PathBuf,
    rootfs: PathBuf,
    scratch: PathBuf,
    console: PathBuf,
}

/// A value only touched on its VM's dispatch queue.
struct OnQueue<T>(T);

// SAFETY: the wrapped Objective-C objects are only used inside closures run
// on the VM's serial queue; other threads only move the wrapper around.
unsafe impl<T> Send for OnQueue<T> {}
// SAFETY: as above; shared access never touches the inner value off-queue.
unsafe impl<T> Sync for OnQueue<T> {}

/// The VM and the objects it needs kept alive.
struct Machine {
    vm: Retained<VZVirtualMachine>,
    _listener: Retained<VZVirtioSocketListener>,
    _delegate: Retained<EgressDelegate>,
}

type Started = std::result::Result<(), String>;

/// A boot failure, reported back through the start channel.
struct BootError(String, oneshot::Sender<Started>);

impl BootError {
    fn report(self) {
        let _ = self.1.send(Err(self.0));
    }
}

impl Machine {
    /// Builds and starts the VM. Runs on the VM's queue.
    fn boot(
        queue: &DispatchQueue,
        paths: &VmPaths,
        cpus: u32,
        memory_bytes: u64,
        egress: mpsc::UnboundedSender<Accepted>,
        started: oneshot::Sender<Started>,
    ) -> std::result::Result<Self, BootError> {
        let config = match configuration(paths, cpus, memory_bytes) {
            Ok(config) => config,
            Err(error) => return Err(BootError(format!("{error:#}"), started)),
        };
        // SAFETY: called on `queue`, the queue the VM is bound to.
        let vm = unsafe {
            VZVirtualMachine::initWithConfiguration_queue(VZVirtualMachine::alloc(), &config, queue)
        };
        let delegate = EgressDelegate::new(egress);
        // SAFETY: plain object construction and wiring on the VM's queue.
        let listener = unsafe {
            let listener = VZVirtioSocketListener::new();
            listener.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
            if let Some(device) = socket_device(&vm) {
                device.setSocketListener_forPort(&listener, EGRESS_PORT);
            }
            listener
        };
        let started = Mutex::new(Some(started));
        let handler = RcBlock::new(move |error: *mut NSError| {
            if let Some(started) = started.lock().ok().and_then(|mut slot| slot.take()) {
                let _ = started.send(match unsafe { error.as_ref() } {
                    None => Ok(()),
                    Some(error) => Err(error.localizedDescription().to_string()),
                });
            }
        });
        // SAFETY: called on the VM's queue; the block outlives the call.
        unsafe { vm.startWithCompletionHandler(&handler) };
        Ok(Self {
            vm,
            _listener: listener,
            _delegate: delegate,
        })
    }
}

fn configuration(
    paths: &VmPaths,
    cpus: u32,
    memory_bytes: u64,
) -> Result<Retained<VZVirtualMachineConfiguration>> {
    // SAFETY: object construction with valid arguments; no shared state.
    unsafe {
        let boot_loader = VZLinuxBootLoader::initWithKernelURL(
            VZLinuxBootLoader::alloc(),
            &file_url(&paths.kernel),
        );
        boot_loader.setCommandLine(&NSString::from_str(&kernel_cmdline("hvc0")));

        let config = VZVirtualMachineConfiguration::new();
        config.setBootLoader(Some(&boot_loader));
        config.setCPUCount(usize::try_from(cpus)?);
        config.setMemorySize(memory_bytes);
        config.setStorageDevices(&NSArray::from_retained_slice(&[
            block_device(&paths.rootfs, true)?,
            block_device(&paths.scratch, false)?,
        ]));
        let socket: Retained<VZSocketDeviceConfiguration> =
            VZVirtioSocketDeviceConfiguration::new().into_super();
        config.setSocketDevices(&NSArray::from_retained_slice(&[socket]));
        let entropy: Retained<VZEntropyDeviceConfiguration> =
            VZVirtioEntropyDeviceConfiguration::new().into_super();
        config.setEntropyDevices(&NSArray::from_retained_slice(&[entropy]));
        config.setSerialPorts(&NSArray::from_retained_slice(&[console_port(
            &paths.console,
        )?]));
        config
            .validateWithError()
            .map_err(|error| anyhow::anyhow!(error.localizedDescription().to_string()))
            .context("the VM configuration is invalid (is the binary signed with the virtualization entitlement?)")?;
        Ok(config)
    }
}

fn block_device(path: &Path, read_only: bool) -> Result<Retained<VZStorageDeviceConfiguration>> {
    // SAFETY: object construction with a valid file URL.
    unsafe {
        let attachment = VZDiskImageStorageDeviceAttachment::initWithURL_readOnly_error(
            VZDiskImageStorageDeviceAttachment::alloc(),
            &file_url(path),
            read_only,
        )
        .map_err(|error| anyhow::anyhow!(error.localizedDescription().to_string()))
        .with_context(|| format!("attach {}", path.display()))?;
        let device = VZVirtioBlockDeviceConfiguration::initWithAttachment(
            VZVirtioBlockDeviceConfiguration::alloc(),
            &attachment,
        );
        Ok(device.into_super())
    }
}

fn console_port(path: &Path) -> Result<Retained<VZSerialPortConfiguration>> {
    // SAFETY: object construction with a valid file URL.
    unsafe {
        let attachment = VZFileSerialPortAttachment::initWithURL_append_error(
            VZFileSerialPortAttachment::alloc(),
            &file_url(path),
            false,
        )
        .map_err(|error| anyhow::anyhow!(error.localizedDescription().to_string()))
        .context("open the console log")?;
        let port = VZVirtioConsoleDeviceSerialPortConfiguration::new();
        port.setAttachment(Some(&attachment));
        Ok(port.into_super())
    }
}

fn file_url(path: &Path) -> Retained<NSURL> {
    NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()))
}

fn socket_device(vm: &VZVirtualMachine) -> Option<Retained<VZVirtioSocketDevice>> {
    // SAFETY: called on the VM's queue; the configuration has one virtio
    // socket device, so the first device is that class.
    unsafe {
        let devices = vm.socketDevices();
        let device = devices.firstObject()?;
        Some(Retained::cast_unchecked(device))
    }
}

/// A guest connection: the socket, plus the framework object that owns the
/// original descriptor, released on the VM's queue when the stream drops.
struct Accepted {
    fd: OwnedFd,
    connection: OnQueue<Retained<VZVirtioSocketConnection>>,
}

fn accept(connection: &VZVirtioSocketConnection) -> Option<Accepted> {
    // SAFETY: the descriptor is valid while `connection` lives; it is
    // duplicated so the stream owns its own descriptor.
    let fd = unsafe { libc::dup(connection.fileDescriptor()) };
    if fd < 0 {
        return None;
    }
    // SAFETY: `dup` returned a new descriptor we now own.
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    Some(Accepted {
        fd,
        connection: OnQueue(connection.retain()),
    })
}

struct EgressIvars {
    egress: mpsc::UnboundedSender<Accepted>,
}

define_class!(
    #[unsafe(super(NSObject))]
    #[name = "MeshComputeEgressListener"]
    #[ivars = EgressIvars]
    struct EgressDelegate;

    unsafe impl NSObjectProtocol for EgressDelegate {}

    unsafe impl VZVirtioSocketListenerDelegate for EgressDelegate {
        #[unsafe(method(listener:shouldAcceptNewConnection:fromSocketDevice:))]
        fn should_accept(
            &self,
            _listener: &VZVirtioSocketListener,
            connection: &VZVirtioSocketConnection,
            _device: &VZVirtioSocketDevice,
        ) -> bool {
            match accept(connection) {
                Some(accepted) => self.ivars().egress.send(accepted).is_ok(),
                None => false,
            }
        }
    }
);

impl EgressDelegate {
    fn new(egress: mpsc::UnboundedSender<Accepted>) -> Retained<Self> {
        let this = Self::alloc().set_ivars(EgressIvars { egress });
        // SAFETY: `NSObject`'s designated initializer.
        unsafe { msg_send![super(this), init] }
    }
}

async fn forward_egress(mut accepted: mpsc::UnboundedReceiver<Accepted>, on_egress: EgressHandler) {
    while let Some(connection) = accepted.recv().await {
        if let Ok(stream) = VzStream::new(connection, None) {
            on_egress(Box::new(stream));
        }
    }
}

pub struct VzVm {
    queue: DispatchRetained<DispatchQueue>,
    machine: Arc<Mutex<Option<OnQueue<Machine>>>>,
    console: PathBuf,
}

#[async_trait]
impl Vm for VzVm {
    async fn connect_agent(&self) -> Result<BoxStream> {
        let (tx, rx) = oneshot::channel::<std::result::Result<Accepted, String>>();
        let machine = self.machine.clone();
        self.queue.exec_async(move || {
            let tx = Mutex::new(Some(tx));
            let reply = move |result| {
                if let Some(tx) = tx.lock().ok().and_then(|mut slot| slot.take()) {
                    let _ = tx.send(result);
                }
            };
            let device = machine.lock().ok().and_then(|slot| {
                slot.as_ref()
                    .and_then(|machine| socket_device(&machine.0.vm))
            });
            let Some(device) = device else {
                reply(Err("the VM is not running".into()));
                return;
            };
            let handler = RcBlock::new(
                move |connection: *mut VZVirtioSocketConnection, error: *mut NSError| {
                    // SAFETY: the framework passes valid pointers or null.
                    let result = match unsafe { (connection.as_ref(), error.as_ref()) } {
                        (Some(connection), _) => {
                            accept(connection).ok_or_else(|| "dup failed".to_string())
                        }
                        (None, Some(error)) => Err(error.localizedDescription().to_string()),
                        (None, None) => Err("no connection".to_string()),
                    };
                    reply(result);
                },
            );
            // SAFETY: called on the VM's queue; the block outlives the call.
            unsafe { device.connectToPort_completionHandler(AGENT_PORT, &handler) };
        });
        let accepted = tokio::time::timeout(CONNECT_TIMEOUT, rx)
            .await
            .context("timed out connecting to the guest agent")?
            .context("the VM queue dropped the connection request")?
            .map_err(|message| anyhow::anyhow!("connect to the guest agent: {message}"))?;
        Ok(Box::new(VzStream::new(accepted, Some(self.queue.clone()))?))
    }

    async fn stop(&self) {
        let (tx, rx) = oneshot::channel::<()>();
        let machine = self.machine.clone();
        self.queue.exec_async(move || {
            let Some(machine) = machine.lock().ok().and_then(|mut slot| slot.take()) else {
                let _ = tx.send(());
                return;
            };
            let vm = machine.0.vm.clone();
            let tx = Mutex::new(Some(tx));
            let keep = Mutex::new(Some(machine));
            let handler = RcBlock::new(move |_error: *mut NSError| {
                // Release the VM here, on its queue, once it has stopped.
                if let Ok(mut keep) = keep.lock() {
                    keep.take();
                }
                if let Some(tx) = tx.lock().ok().and_then(|mut slot| slot.take()) {
                    let _ = tx.send(());
                }
            });
            // SAFETY: called on the VM's queue; the block outlives the call.
            unsafe { vm.stopWithCompletionHandler(&handler) };
        });
        let _ = tokio::time::timeout(STOP_TIMEOUT, rx).await;
    }

    fn console_log(&self) -> Option<PathBuf> {
        Some(self.console.clone())
    }
}

/// A vsock connection as a tokio stream.
struct VzStream {
    inner: UnixStream,
    connection: Option<OnQueue<Retained<VZVirtioSocketConnection>>>,
    queue: Option<DispatchRetained<DispatchQueue>>,
}

impl VzStream {
    fn new(accepted: Accepted, queue: Option<DispatchRetained<DispatchQueue>>) -> Result<Self> {
        let std_stream = std::os::unix::net::UnixStream::from(accepted.fd);
        std_stream.set_nonblocking(true)?;
        Ok(Self {
            inner: UnixStream::from_std(std_stream)?,
            connection: Some(accepted.connection),
            queue,
        })
    }
}

impl Drop for VzStream {
    fn drop(&mut self) {
        if let (Some(connection), Some(queue)) = (self.connection.take(), self.queue.as_ref()) {
            queue.exec_async(move || drop(connection));
        }
    }
}

impl AsyncRead for VzStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for VzStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}
