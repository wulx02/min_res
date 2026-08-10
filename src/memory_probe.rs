use std::env;
use std::sync::mpsc;
use std::time::Duration;

#[derive(Clone, Copy, Debug)]
pub struct ProcessMemory {
    pub resident_bytes: u64,
    pub resident_peak_bytes: u64,
    pub compressed_bytes: u64,
    pub compressed_peak_bytes: u64,
    pub phys_footprint_bytes: u64,
    pub phys_footprint_peak_bytes: i64,
}

impl ProcessMemory {
    pub fn now() -> Option<Self> {
        process_memory_now()
    }
}

pub struct MemoryMonitorGuard {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl MemoryMonitorGuard {
    pub fn start_from_env() -> Option<Self> {
        let interval = env::var("NASSAU_PROCESS_MEMORY_INTERVAL_SECS")
            .ok()
            .and_then(|raw| raw.parse::<u64>().ok())
            .filter(|&seconds| seconds > 0)?;
        ProcessMemory::now()?;

        let (stop, stop_rx) = mpsc::channel::<()>();
        let thread = std::thread::spawn(move || {
            let interval = Duration::from_secs(interval);
            loop {
                if let Some(memory) = ProcessMemory::now() {
                    eprintln!(
                        "memory nassau_min_res sample rss_gib={:.3} rss_peak_gib={:.3} compressed_gib={:.3} compressed_peak_gib={:.3} phys_footprint_gib={:.3} phys_footprint_peak_gib={:.3}",
                        bytes_to_gib(memory.resident_bytes),
                        bytes_to_gib(memory.resident_peak_bytes),
                        bytes_to_gib(memory.compressed_bytes),
                        bytes_to_gib(memory.compressed_peak_bytes),
                        bytes_to_gib(memory.phys_footprint_bytes),
                        bytes_to_gib_signed(memory.phys_footprint_peak_bytes),
                    );
                }
                if stop_rx.recv_timeout(interval).is_ok() {
                    break;
                }
            }
        });

        Some(Self {
            stop: Some(stop),
            thread: Some(thread),
        })
    }
}

impl Drop for MemoryMonitorGuard {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub fn log_process_memory(phase: &str, details: impl AsRef<str>) {
    if env::var_os("NASSAU_PROFILE_MEMORY").is_none() {
        return;
    }
    let details = details.as_ref();
    if let Some(memory) = ProcessMemory::now() {
        eprintln!(
            "nassau_mem_process phase={phase} {details} rss_gib={:.3} rss_peak_gib={:.3} compressed_gib={:.3} compressed_peak_gib={:.3} phys_footprint_gib={:.3} phys_footprint_peak_gib={:.3}",
            bytes_to_gib(memory.resident_bytes),
            bytes_to_gib(memory.resident_peak_bytes),
            bytes_to_gib(memory.compressed_bytes),
            bytes_to_gib(memory.compressed_peak_bytes),
            bytes_to_gib(memory.phys_footprint_bytes),
            bytes_to_gib_signed(memory.phys_footprint_peak_bytes),
        );
    }
}

#[cfg(target_os = "macos")]
fn process_memory_now() -> Option<ProcessMemory> {
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct TaskVmInfo {
        virtual_size: u64,
        region_count: i32,
        page_size: i32,
        resident_size: u64,
        resident_size_peak: u64,
        device: u64,
        device_peak: u64,
        internal: u64,
        internal_peak: u64,
        external: u64,
        external_peak: u64,
        reusable: u64,
        reusable_peak: u64,
        purgeable_volatile_pmap: u64,
        purgeable_volatile_resident: u64,
        purgeable_volatile_virtual: u64,
        compressed: u64,
        compressed_peak: u64,
        compressed_lifetime: u64,
        phys_footprint: u64,
        min_address: u64,
        max_address: u64,
        ledger_phys_footprint_peak: i64,
    }

    const TASK_VM_INFO: u32 = 22;
    const KERN_SUCCESS: i32 = 0;

    unsafe extern "C" {
        static mach_task_self_: libc::c_uint;
        fn task_info(
            target_task: libc::c_uint,
            flavor: libc::c_uint,
            task_info_out: *mut libc::c_int,
            task_info_out_count: *mut libc::c_uint,
        ) -> libc::c_int;
    }

    let mut info = std::mem::MaybeUninit::<TaskVmInfo>::zeroed();
    let mut count =
        (std::mem::size_of::<TaskVmInfo>() / std::mem::size_of::<libc::c_uint>()) as libc::c_uint;
    let result = unsafe {
        task_info(
            mach_task_self_,
            TASK_VM_INFO,
            info.as_mut_ptr().cast::<libc::c_int>(),
            &mut count,
        )
    };
    if result != KERN_SUCCESS {
        return None;
    }
    let info = unsafe { info.assume_init() };
    Some(ProcessMemory {
        resident_bytes: info.resident_size,
        resident_peak_bytes: info.resident_size_peak,
        compressed_bytes: info.compressed,
        compressed_peak_bytes: info.compressed_peak,
        phys_footprint_bytes: info.phys_footprint,
        phys_footprint_peak_bytes: info.ledger_phys_footprint_peak,
    })
}

#[cfg(target_os = "linux")]
fn process_memory_now() -> Option<ProcessMemory> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let resident_bytes = parse_status_kib(&status, "VmRSS:")?.saturating_mul(1024);
    let resident_peak_bytes = parse_status_kib(&status, "VmHWM:")
        .unwrap_or(resident_bytes / 1024)
        .saturating_mul(1024);
    let swap_bytes = parse_status_kib(&status, "VmSwap:")
        .unwrap_or(0)
        .saturating_mul(1024);
    Some(ProcessMemory {
        resident_bytes,
        resident_peak_bytes,
        compressed_bytes: 0,
        compressed_peak_bytes: 0,
        phys_footprint_bytes: resident_bytes.saturating_add(swap_bytes),
        phys_footprint_peak_bytes: resident_peak_bytes.saturating_add(swap_bytes) as i64,
    })
}

#[cfg(target_os = "linux")]
fn parse_status_kib(status: &str, key: &str) -> Option<u64> {
    let line = status.lines().find(|line| line.starts_with(key))?;
    line.split_whitespace().nth(1)?.parse::<u64>().ok()
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn process_memory_now() -> Option<ProcessMemory> {
    None
}

pub fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

pub fn bytes_to_gib_signed(bytes: i64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}
