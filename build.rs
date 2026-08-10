use std::env;
use std::fs;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=GIT_COMMIT");
    emit_git_rerun_paths();

    if let Ok(commit) = env::var("GIT_COMMIT") {
        println!("cargo:rustc-env=GIT_COMMIT={commit}");
        return;
    }

    let Some(mut commit) = git_output(&["rev-parse", "--short=12", "HEAD"]) else {
        println!("cargo:rustc-env=GIT_COMMIT=unknown");
        return;
    };
    if git_is_dirty() {
        commit.push_str("-dirty");
    }
    println!("cargo:rustc-env=GIT_COMMIT={commit}");
}

fn git_output(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn git_is_dirty() -> bool {
    let status = Command::new("git")
        .args(["diff", "--quiet", "--ignore-submodules", "HEAD"])
        .status();
    !matches!(status, Ok(status) if status.success())
}

fn emit_git_rerun_paths() {
    println!("cargo:rerun-if-changed=.git/HEAD");
    let Ok(head) = fs::read_to_string(".git/HEAD") else {
        return;
    };
    let Some(ref_path) = head.trim().strip_prefix("ref: ") else {
        return;
    };
    println!("cargo:rerun-if-changed=.git/{ref_path}");
}
