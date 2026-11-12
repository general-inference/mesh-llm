//! File reads, writes and listings.

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::Path;

use anyhow::{Context, Result, bail};

use super::transport::Conn;
use crate::protocol::{DirEntry, EntryKind, Reply, Request, read_frame};

pub fn read(path: &str, max_bytes: u64) -> Result<Reply> {
    let file = fs::File::open(path).with_context(|| format!("open {path}"))?;
    let size = file.metadata()?.len();
    let mut data = Vec::new();
    file.take(max_bytes).read_to_end(&mut data)?;
    Ok(Reply::File {
        truncated: size > data.len() as u64,
        data,
        size,
    })
}

pub fn write(path: &str, mode: Option<u32>, append: bool, conn: &mut Box<dyn Conn>) -> Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut options = OpenOptions::new();
    options.write(true).create(true);
    if append {
        options.append(true);
    } else {
        options.truncate(true);
    }
    if let Some(mode) = mode {
        options.mode(mode);
    }
    let mut file = options.open(path).with_context(|| format!("open {path}"))?;
    copy_chunks(conn, &mut file)?;
    if let Some(mode) = mode {
        fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
    }
    Ok(())
}

pub fn list(path: &str) -> Result<Reply> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(path).with_context(|| format!("list {path}"))? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        let file_type = metadata.file_type();
        let kind = if file_type.is_symlink() {
            EntryKind::Symlink
        } else if file_type.is_dir() {
            EntryKind::Dir
        } else if file_type.is_file() {
            EntryKind::File
        } else {
            EntryKind::Other
        };
        entries.push(DirEntry {
            name: entry.file_name().to_string_lossy().into_owned(),
            kind,
            size: metadata.len(),
        });
    }
    entries.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(Reply::Entries { entries })
}

/// Copies `Chunk` frames into `sink` until `End`.
pub fn copy_chunks(conn: &mut Box<dyn Conn>, sink: &mut impl Write) -> Result<()> {
    loop {
        match read_frame::<Request>(conn)? {
            Some(Request::Chunk { data }) => sink.write_all(&data)?,
            Some(Request::End) => return Ok(()),
            Some(_) => bail!("expected a chunk or end frame"),
            None => bail!("connection closed before the end frame"),
        }
    }
}
