use duckdb::{params, Connection};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{exit, Command, ExitStatus};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const SPIKE_TABLE: &str = "_photolib_h1_handoff_spike";

fn main()
{
    let args: Vec<String> = env::args().collect();
    let result = match args.get(1).map(String::as_str)
    {
        Some("run") => run_orchestrator(&args),
        Some("agent") => run_agent(&args),
        Some("open-once") => run_open_once(&args),
        _ =>
        {
            print_usage(&args);
            Err("invalid arguments".to_string())
        }
    };

    if let Err(message) = result
    {
        eprintln!("duckdb_handoff_spike: {}", message);
        exit(1);
    }
}

fn print_usage(args: &[String])
{
    let program = args
        .first()
        .map(String::as_str)
        .unwrap_or("duckdb_handoff_spike");
    eprintln!("Usage:");
    eprintln!("  {} run <catalogue-path>", program);
    eprintln!(
        "  {} agent <catalogue-path> <ready-file> <release-file>",
        program
    );
    eprintln!("  {} open-once <catalogue-path> <phase>", program);
}

fn run_orchestrator(args: &[String]) -> Result<(), String>
{
    let catalogue_path = args
        .get(2)
        .map(PathBuf::from)
        .ok_or_else(|| "run requires <catalogue-path>".to_string())?;

    if let Some(parent) = catalogue_path.parent()
    {
        fs::create_dir_all(parent)
            .map_err(|e| format!("create catalogue parent failed: {}", e))?;
    }

    let control_dir = make_control_dir()?;
    let ready_file = control_dir.join("agent.ready");
    let release_file = control_dir.join("agent.release");
    let current_exe = env::current_exe()
        .map_err(|e| format!("current_exe failed: {}", e))?;

    println!("H1 spike catalogue: {}", catalogue_path.display());
    println!("H1 spike control: {}", control_dir.display());

    let mut agent = Command::new(&current_exe)
        .arg("agent")
        .arg(&catalogue_path)
        .arg(&ready_file)
        .arg(&release_file)
        .spawn()
        .map_err(|e| format!("spawn agent failed: {}", e))?;

    wait_for_file(&ready_file, Duration::from_secs(10))
        .map_err(|e|
        {
            let _ = agent.kill();
            e
        })?;

    println!("agent has catalogue open read-write");

    match open_and_write(&catalogue_path, "app_overlapping_writer_probe")
    {
        Ok(()) =>
        {
            let _ = fs::write(&release_file, b"release");
            let _ = agent.wait();
            return Err(
                "overlapping app writer unexpectedly opened and wrote while agent held catalogue"
                    .to_string(),
            );
        }
        Err(e) =>
        {
            println!("overlapping app writer blocked as expected: {}", e);
        }
    }

    fs::write(&release_file, b"release")
        .map_err(|e| format!("write release file failed: {}", e))?;

    let status = agent
        .wait()
        .map_err(|e| format!("wait for agent failed: {}", e))?;
    ensure_success(status, "agent")?;

    println!("agent released catalogue");

    open_and_write(&catalogue_path, "app_after_release")
        .map_err(|e| format!("app open after agent release failed: {}", e))?;

    let event_count = count_spike_events(&catalogue_path)?;
    println!(
        "app reopened and wrote after release; spike event count={}",
        event_count
    );
    println!("H1 spike PASS");

    Ok(())
}

fn run_agent(args: &[String]) -> Result<(), String>
{
    let catalogue_path = args
        .get(2)
        .map(PathBuf::from)
        .ok_or_else(|| "agent requires <catalogue-path>".to_string())?;
    let ready_file = args
        .get(3)
        .map(PathBuf::from)
        .ok_or_else(|| "agent requires <ready-file>".to_string())?;
    let release_file = args
        .get(4)
        .map(PathBuf::from)
        .ok_or_else(|| "agent requires <release-file>".to_string())?;

    let conn = Connection::open(&catalogue_path)
        .map_err(|e| format!("agent open failed: {}", e))?;
    ensure_spike_table(&conn)?;
    insert_spike_event(&conn, "agent_open")?;

    fs::write(&ready_file, b"ready")
        .map_err(|e| format!("agent write ready failed: {}", e))?;

    while !release_file.exists()
    {
        thread::sleep(Duration::from_millis(100));
    }

    insert_spike_event(&conn, "agent_release_requested")?;
    conn.execute_batch("CHECKPOINT;")
        .map_err(|e| format!("agent checkpoint failed: {}", e))?;

    drop(conn);
    Ok(())
}

fn run_open_once(args: &[String]) -> Result<(), String>
{
    let catalogue_path = args
        .get(2)
        .map(PathBuf::from)
        .ok_or_else(|| "open-once requires <catalogue-path>".to_string())?;
    let phase = args
        .get(3)
        .map(String::as_str)
        .ok_or_else(|| "open-once requires <phase>".to_string())?;

    open_and_write(&catalogue_path, phase)
}

fn open_and_write(catalogue_path: &Path, phase: &str) -> Result<(), String>
{
    let conn = Connection::open(catalogue_path)
        .map_err(|e| format!("open failed: {}", e))?;
    ensure_spike_table(&conn)?;
    insert_spike_event(&conn, phase)?;
    conn.execute_batch("CHECKPOINT;")
        .map_err(|e| format!("checkpoint failed: {}", e))?;
    drop(conn);
    Ok(())
}

fn ensure_spike_table(conn: &Connection) -> Result<(), String>
{
    let sql = format!(
        "CREATE TABLE IF NOT EXISTS {} (
            phase TEXT NOT NULL,
            detail TEXT,
            pid BIGINT,
            event_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );",
        SPIKE_TABLE
    );
    conn.execute_batch(&sql)
        .map_err(|e| format!("ensure spike table failed: {}", e))
}

fn insert_spike_event(conn: &Connection, phase: &str) -> Result<(), String>
{
    let sql = format!(
        "INSERT INTO {} (phase, detail, pid) VALUES (?1, ?2, ?3)",
        SPIKE_TABLE
    );
    conn.execute(
        &sql,
        params![
            phase,
            "duckdb handoff spike",
            i64::from(std::process::id())
        ],
    )
    .map(|_| ())
    .map_err(|e| format!("insert spike event '{}' failed: {}", phase, e))
}

fn count_spike_events(catalogue_path: &Path) -> Result<i64, String>
{
    let conn = Connection::open(catalogue_path)
        .map_err(|e| format!("count open failed: {}", e))?;
    let sql = format!("SELECT COUNT(*) FROM {}", SPIKE_TABLE);
    conn.query_row(&sql, [], |row| row.get::<_, i64>(0))
        .map_err(|e| format!("count query failed: {}", e))
}

fn make_control_dir() -> Result<PathBuf, String>
{
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("system clock before epoch: {}", e))?
        .as_millis();
    let dir = env::temp_dir().join(format!(
        "photolib_h1_handoff_{}_{}",
        std::process::id(),
        now
    ));
    fs::create_dir_all(&dir)
        .map_err(|e| format!("create control dir failed: {}", e))?;
    Ok(dir)
}

fn wait_for_file(path: &Path, timeout: Duration) -> Result<(), String>
{
    let start = Instant::now();
    while start.elapsed() < timeout
    {
        if path.exists()
        {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(50));
    }
    Err(format!("timed out waiting for {}", path.display()))
}

fn ensure_success(status: ExitStatus, label: &str) -> Result<(), String>
{
    if status.success()
    {
        Ok(())
    }
    else
    {
        Err(format!("{} exited with {}", label, status))
    }
}
