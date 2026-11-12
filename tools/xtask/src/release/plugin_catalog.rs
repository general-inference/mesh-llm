//! `release plugin-catalog-pin`: pins an in-repo plugin's catalog entry to the
//! release just published, so `mesh-llm plugins install <name>` installs
//! exactly those archives.
//!
//! ```text
//! release plugin-catalog-pin --entry <entry.json> --tag <tag>
//!     --archives <dir> --catalog <plugins.jsonl> --out <plugins.jsonl>
//!     [--hf-commit <payload.ndjson> --parent <commit-sha>]
//! ```
//!
//! The entry's static fields come from the checked-in `entry.json`; `version`
//! becomes the tag and `sha256` the digest of each `<name>-<tag>-<target>`
//! archive in `--archives`. The catalog line with the same `name` is replaced
//! (or the entry appended); every other line is copied byte for byte. A
//! missing `--catalog` file is an empty catalog.
//!
//! With `--hf-commit`, it also writes the body of a Hugging Face commit-API
//! request that replaces `plugins.jsonl`, conditioned on `--parent` (the
//! revision `--catalog` was read at): if anyone changed the catalog since, the
//! commit is refused rather than overwriting their change.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::repository::check_report::CheckReport;

const USAGE: &str = "usage: release plugin-catalog-pin --entry <entry.json> --tag <tag> \
--archives <dir> --catalog <plugins.jsonl> --out <plugins.jsonl> \
[--hf-commit <payload.ndjson> --parent <commit-sha>]";
const FLAGS: [&str; 7] = [
    "entry",
    "tag",
    "archives",
    "catalog",
    "out",
    "hf-commit",
    "parent",
];
const ARCHIVE_EXTENSIONS: [&str; 2] = [".tar.gz", ".zip"];

pub(super) fn run(args: &[String]) -> CheckReport {
    match parse(args).and_then(|options| pin(&options)) {
        Ok(line) => CheckReport {
            stdout: format!("{line}\n"),
            ..Default::default()
        },
        Err(error) => CheckReport {
            stderr: format!("{error}\n"),
            code: 1,
            ..Default::default()
        },
    }
}

struct Options {
    entry: PathBuf,
    tag: String,
    archives: PathBuf,
    catalog: PathBuf,
    out: PathBuf,
    hf_commit: Option<(PathBuf, String)>,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut values: BTreeMap<&str, &str> = BTreeMap::new();
    let mut iter = args.iter();
    while let Some(flag) = iter.next() {
        let key = flag
            .strip_prefix("--")
            .filter(|key| FLAGS.contains(key))
            .ok_or_else(|| format!("unknown argument `{flag}`\n{USAGE}"))?;
        let value = iter
            .next()
            .ok_or_else(|| format!("--{key} needs a value"))?;
        if values.insert(key, value).is_some() {
            return Err(format!("--{key} given twice"));
        }
    }
    let take = |key: &str| {
        values
            .get(key)
            .map(|value| (*value).to_string())
            .ok_or_else(|| format!("--{key} is required\n{USAGE}"))
    };
    let tag = take("tag")?;
    if tag.is_empty() || !tag.starts_with('v') || tag.contains(['/', '\\', ' ']) {
        return Err(format!("`{tag}` is not a release tag"));
    }
    let hf_commit = match (values.get("hf-commit"), values.get("parent")) {
        (None, None) => None,
        (Some(path), Some(parent)) => {
            if parent.len() != 40 || !parent.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(format!("`{parent}` is not a commit sha"));
            }
            Some((PathBuf::from(*path), (*parent).to_string()))
        }
        _ => return Err("--hf-commit and --parent go together".into()),
    };
    Ok(Options {
        entry: take("entry")?.into(),
        tag,
        archives: take("archives")?.into(),
        catalog: take("catalog")?.into(),
        out: take("out")?.into(),
        hf_commit,
    })
}

fn pin(options: &Options) -> Result<String, String> {
    let entry_text = read(&options.entry)?;
    let mut entry: Map<String, Value> = serde_json::from_str(&entry_text)
        .map_err(|error| format!("{} is not a JSON object: {error}", options.entry.display()))?;
    let name = entry
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| format!("{} has no name", options.entry.display()))?
        .to_string();
    let digests = archive_digests(&options.archives, &name, &options.tag)?;
    entry.insert("version".into(), Value::String(options.tag.clone()));
    entry.insert(
        "sha256".into(),
        Value::Object(
            digests
                .into_iter()
                .map(|(target, digest)| (target, Value::String(digest)))
                .collect(),
        ),
    );
    let line = serde_json::to_string(&entry).map_err(|error| error.to_string())?;
    let catalog = match std::fs::read_to_string(&options.catalog) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => return Err(format!("read {}: {error}", options.catalog.display())),
    };
    let updated = replace_line(&catalog, &name, &line)?;
    std::fs::write(&options.out, &updated)
        .map_err(|error| format!("write {}: {error}", options.out.display()))?;
    if let Some((path, parent)) = &options.hf_commit {
        let payload =
            hf_commit_payload(&updated, &format!("Pin {name} to {}", options.tag), parent);
        std::fs::write(path, payload)
            .map_err(|error| format!("write {}: {error}", path.display()))?;
    }
    Ok(line)
}

/// NDJSON body for `POST /api/datasets/<repo>/commit/main`.
fn hf_commit_payload(catalog: &str, summary: &str, parent: &str) -> String {
    use base64::Engine as _;
    let header = serde_json::json!({
        "key": "header",
        "value": { "summary": summary, "description": "", "parentCommit": parent },
    });
    let file = serde_json::json!({
        "key": "file",
        "value": {
            "path": "plugins.jsonl",
            "encoding": "base64",
            "content": base64::engine::general_purpose::STANDARD.encode(catalog),
        },
    });
    format!("{header}\n{file}\n")
}

/// `target -> sha256` for every `<name>-<tag>-<target><ext>` in `dir`.
fn archive_digests(dir: &Path, name: &str, tag: &str) -> Result<BTreeMap<String, String>, String> {
    let prefix = format!("{name}-{tag}-");
    let mut digests = BTreeMap::new();
    let entries =
        std::fs::read_dir(dir).map_err(|error| format!("read {}: {error}", dir.display()))?;
    for entry in entries.flatten() {
        let file_name = entry.file_name().to_string_lossy().into_owned();
        let Some(rest) = file_name.strip_prefix(&prefix) else {
            continue;
        };
        let Some(target) = ARCHIVE_EXTENSIONS
            .iter()
            .find_map(|extension| rest.strip_suffix(extension))
        else {
            continue;
        };
        let bytes =
            std::fs::read(entry.path()).map_err(|error| format!("read {file_name}: {error}"))?;
        digests.insert(target.to_string(), hex::encode(Sha256::digest(&bytes)));
    }
    if digests.is_empty() {
        return Err(format!(
            "no `{prefix}<target>` archives in {}",
            dir.display()
        ));
    }
    Ok(digests)
}

fn replace_line(catalog: &str, name: &str, line: &str) -> Result<String, String> {
    let mut out = String::with_capacity(catalog.len() + line.len() + 1);
    let mut replaced = false;
    for (index, existing) in catalog.lines().enumerate() {
        let same_name = !existing.trim().is_empty()
            && serde_json::from_str::<Value>(existing)
                .map_err(|error| format!("catalog line {} is not JSON: {error}", index + 1))?
                .get("name")
                .and_then(Value::as_str)
                == Some(name);
        if same_name {
            if replaced {
                return Err(format!("the catalog lists `{name}` more than once"));
            }
            out.push_str(line);
            replaced = true;
        } else {
            out.push_str(existing);
        }
        out.push('\n');
    }
    if !replaced {
        out.push_str(line);
        out.push('\n');
    }
    Ok(out)
}

fn read(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|error| format!("read {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENTRY: &str = r#"{"name":"sandbox","description":"Sandboxes","github_url":"https://github.com/Mesh-LLM/mesh-llm","author_email":"m@example.com","author_name":"Mesh LLM"}"#;

    fn options(dir: &Path, catalog: &str) -> Options {
        std::fs::write(dir.join("entry.json"), ENTRY).unwrap();
        let archives = dir.join("archives");
        std::fs::create_dir_all(&archives).unwrap();
        std::fs::write(
            archives.join("sandbox-v1.2.0-x86_64-unknown-linux-gnu.tar.gz"),
            b"a",
        )
        .unwrap();
        std::fs::write(
            archives.join("sandbox-v1.2.0-x86_64-pc-windows-msvc.zip"),
            b"b",
        )
        .unwrap();
        std::fs::write(
            archives.join("sandbox-image-base-v1.2.0-x86_64.tar.gz"),
            b"c",
        )
        .unwrap();
        std::fs::write(
            archives.join("sandbox-v1.1.0-aarch64-apple-darwin.tar.gz"),
            b"d",
        )
        .unwrap();
        std::fs::write(dir.join("plugins.jsonl"), catalog).unwrap();
        Options {
            entry: dir.join("entry.json"),
            tag: "v1.2.0".into(),
            archives,
            catalog: dir.join("plugins.jsonl"),
            out: dir.join("out.jsonl"),
            hf_commit: None,
        }
    }

    #[test]
    fn pins_this_releases_archives_and_keeps_other_lines() {
        let dir = tempfile::tempdir().unwrap();
        let other =
            r#"{"name":"blackboard","github_url":"https://github.com/mesh-llm/blackboard"}"#;
        let stale = r#"{"name":"sandbox","version":"v1.0.0"}"#;
        let options = options(dir.path(), &format!("{other}\n{stale}\n"));
        pin(&options).unwrap();
        let written = std::fs::read_to_string(&options.out).unwrap();
        let lines: Vec<&str> = written.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], other);
        let entry: Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(entry["version"], "v1.2.0");
        assert_eq!(entry["description"], "Sandboxes");
        let sha256 = entry["sha256"].as_object().unwrap();
        assert_eq!(
            sha256.len(),
            2,
            "only this release's plugin archives: {sha256:?}"
        );
        assert_eq!(
            sha256["x86_64-unknown-linux-gnu"],
            hex::encode(Sha256::digest(b"a"))
        );
        assert!(sha256.contains_key("x86_64-pc-windows-msvc"));
    }

    #[test]
    fn a_new_plugin_is_appended_to_a_missing_catalog() {
        let dir = tempfile::tempdir().unwrap();
        let mut options = options(dir.path(), "");
        options.catalog = dir.path().join("absent.jsonl");
        pin(&options).unwrap();
        let written = std::fs::read_to_string(&options.out).unwrap();
        assert_eq!(written.lines().count(), 1);
    }

    #[test]
    fn the_hf_commit_replaces_the_catalog_only_on_top_of_its_parent() {
        let dir = tempfile::tempdir().unwrap();
        let mut options = options(dir.path(), "");
        let parent = "0d072047434832ac93f2c8bb3f03dc3a025db0ad".to_string();
        options.hf_commit = Some((dir.path().join("commit.ndjson"), parent.clone()));
        pin(&options).unwrap();
        let payload = std::fs::read_to_string(dir.path().join("commit.ndjson")).unwrap();
        let lines: Vec<Value> = payload
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines[0]["value"]["parentCommit"], parent.as_str());
        assert_eq!(lines[0]["value"]["summary"], "Pin sandbox to v1.2.0");
        use base64::Engine as _;
        let content = base64::engine::general_purpose::STANDARD
            .decode(lines[1]["value"]["content"].as_str().unwrap())
            .unwrap();
        assert_eq!(
            String::from_utf8(content).unwrap(),
            std::fs::read_to_string(&options.out).unwrap()
        );
        let flags = |extra: &[&str]| {
            ["--entry", "e", "--tag", "v1.0.0", "--archives", "a"]
                .iter()
                .chain(["--catalog", "c", "--out", "o"].iter())
                .chain(extra)
                .map(ToString::to_string)
                .collect::<Vec<String>>()
        };
        assert!(parse(&flags(&["--hf-commit", "p"])).is_err());
        assert!(parse(&flags(&["--hf-commit", "p", "--parent", "main"])).is_err());
    }

    #[test]
    fn refuses_bad_tags_missing_archives_and_duplicate_entries() {
        assert!(parse(&["--tag".into(), "1.2.0".into()]).is_err());
        let dir = tempfile::tempdir().unwrap();
        let mut options = options(dir.path(), "");
        options.tag = "v9.9.9".into();
        assert!(pin(&options).unwrap_err().contains("no `sandbox-v9.9.9-"));
        let twice = "{\"name\":\"sandbox\"}\n{\"name\":\"sandbox\"}\n";
        assert!(replace_line(twice, "sandbox", "{}").is_err());
    }
}
