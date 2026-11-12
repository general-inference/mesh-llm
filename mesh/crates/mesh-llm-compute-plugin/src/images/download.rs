//! Downloads a published image and installs it once its digest matches.
//!
//! The archive is hashed as it streams to disk and refused unless it matches
//! the compiled-in pin; only then is it unpacked, into a staging directory
//! renamed into place, so a crash or a bad download never leaves a half
//! image where the store would find it.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use super::pins::ImagePin;
use super::{KERNEL_FILE, ROOTFS_FILE};

/// The archive is about a quarter of a gigabyte; this bounds a stalled
/// connection, not a slow one.
const READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

pub async fn install(root: &Path, base_url: &str, pin: &ImagePin) -> Result<()> {
    let tag = super::pins::release_tag();
    let url = format!(
        "{}/{tag}/{}",
        base_url.trim_end_matches('/'),
        pin.asset_name(&tag)
    );
    std::fs::create_dir_all(root)?;
    let staging = root.join(format!(".staging-{}-{}", pin.name, pin.arch));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging)?;
    let result = fetch_and_unpack(&url, &staging, pin).await;
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    result?;
    let target = root.join(&pin.name).join(&pin.arch);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let _ = std::fs::remove_dir_all(&target);
    std::fs::rename(staging.join("image"), &target)?;
    let _ = std::fs::remove_dir_all(&staging);
    Ok(())
}

async fn fetch_and_unpack(url: &str, staging: &Path, pin: &ImagePin) -> Result<()> {
    let archive = staging.join("image.tar.gz");
    let digest = fetch(url, &archive).await?;
    if digest != pin.sha256 {
        bail!(
            "{url} has sha256 {digest}, but this build expects {}; refusing it",
            pin.sha256
        );
    }
    let unpacked = staging.join("image");
    tokio::task::spawn_blocking({
        let unpacked = unpacked.clone();
        move || unpack(&archive, &unpacked)
    })
    .await??;
    for file in [KERNEL_FILE, ROOTFS_FILE] {
        if !unpacked.join(file).is_file() {
            bail!("{url} does not contain {file}");
        }
    }
    Ok(())
}

/// Streams `url` to `path` and returns the sha256 of what was written.
async fn fetch(url: &str, path: &PathBuf) -> Result<String> {
    let mut builder = reqwest::Client::builder().read_timeout(READ_TIMEOUT);
    if url.starts_with("http://") {
        // A plain-HTTP mirror never needs TLS; trusting no roots means a host
        // without a CA bundle can still use one.
        builder = builder.tls_certs_only([]);
    }
    let client = builder
        .build()
        .context("set up downloads (an https image_url needs the system's CA certificates)")?;
    let mut response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("download {url}"))?
        .error_for_status()
        .with_context(|| format!("download {url}"))?;
    let mut file = std::fs::File::create(path)?;
    let mut hasher = Sha256::new();
    while let Some(chunk) = response.chunk().await? {
        hasher.update(&chunk);
        file.write_all(&chunk)?;
    }
    file.sync_all()?;
    Ok(hex::encode(hasher.finalize()))
}

fn unpack(archive: &Path, into: &Path) -> Result<()> {
    std::fs::create_dir_all(into)?;
    let file = std::fs::File::open(archive)?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
    for entry in tar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        // Only the files an image has, at the top level, as regular files.
        let wanted = matches!(name, KERNEL_FILE | ROOTFS_FILE | "image.json")
            && path.components().count() == 1
            && entry.header().entry_type().is_file();
        if wanted {
            entry.unpack(into.join(name))?;
        }
    }
    let _ = std::fs::remove_file(archive);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn archive_with(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut builder = tar::Builder::new(flate2::write::GzEncoder::new(
            Vec::new(),
            flate2::Compression::fast(),
        ));
        for (name, data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, name, *data).unwrap();
        }
        builder.into_inner().unwrap().finish().unwrap()
    }

    #[test]
    fn unpacking_keeps_only_top_level_image_files() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("image.tar.gz");
        std::fs::write(
            &archive,
            archive_with(&[
                ("kernel", b"k"),
                ("rootfs.ext4", b"r"),
                ("nested/kernel", b"x"),
                ("extra", b"x"),
            ]),
        )
        .unwrap();
        let into = dir.path().join("image");
        unpack(&archive, &into).unwrap();
        let mut names: Vec<String> = std::fs::read_dir(&into)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(names, ["kernel", "rootfs.ext4"]);
        assert_eq!(std::fs::read(into.join("kernel")).unwrap(), b"k");
    }

    /// Serves `body` to one request and returns the base URL.
    async fn serve_once(body: Vec<u8>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 1024];
            let _ = socket.read(&mut request).await;
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            socket.write_all(head.as_bytes()).await.unwrap();
            socket.write_all(&body).await.unwrap();
        });
        format!("http://{address}")
    }

    fn pin(sha256: String) -> ImagePin {
        ImagePin {
            name: "base".into(),
            arch: "x86_64".into(),
            sha256,
        }
    }

    #[tokio::test]
    async fn a_matching_download_is_installed() {
        let body = archive_with(&[("kernel", b"k"), ("rootfs.ext4", b"r")]);
        let digest = hex::encode(Sha256::digest(&body));
        let base = serve_once(body).await;
        let root = tempfile::tempdir().unwrap();
        install(root.path(), &base, &pin(digest)).await.unwrap();
        let image = root.path().join("base").join("x86_64");
        assert_eq!(std::fs::read(image.join("rootfs.ext4")).unwrap(), b"r");
        assert!(!root.path().join(".staging-base-x86_64").exists());
    }

    #[tokio::test]
    async fn a_digest_mismatch_installs_nothing() {
        let base = serve_once(archive_with(&[("kernel", b"k"), ("rootfs.ext4", b"r")])).await;
        let root = tempfile::tempdir().unwrap();
        let error = install(root.path(), &base, &pin("0".repeat(64)))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("refusing"), "{error}");
        assert!(!root.path().join("base").exists());
        assert!(!root.path().join(".staging-base-x86_64").exists());
    }
}
