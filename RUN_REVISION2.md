# Running the revision (short guide)

Everything is one script that you can stop and start again at any point. Closing the
laptop lid, sleep, moving between college and home, even a power cut: none of it loses
more than the last few minutes. Finished days and runs are always kept.

## One-time setup

1. Put `fl-ids-revision2.zip` in your Downloads folder (same place as before).
2. Open WSL and run these three lines:

```bash
mkdir -p ~/rev2_work
cd ~/rev2_work
unzip -o /mnt/c/Users/Dell/Downloads/fl-ids-revision2.zip
```

That unpacks the code to `~/rev2_work/rev2`.

## Start it

```bash
bash ~/rev2_work/rev2/run_revision2.sh start
```

It runs in the background, so you can close the terminal. It picks up where it left off
every time you run `start`, so this is also the command you use to resume after a pause.

## Check how far along it is

```bash
bash ~/rev2_work/rev2/run_revision2.sh status
```

This prints the current step, how many runs are done, a rough time left, and free disk
space. The last few lines of progress are also in `~/rev2_work/STATUS.txt`.

## Pause before you move or shut down

```bash
bash ~/rev2_work/rev2/run_revision2.sh stop
```

It finishes the small piece it is on, saves, and exits. After that you can shut down or
close the lid. To continue later, run `start` again.

You do not strictly need to run `stop` first: if the battery dies or you just close the
lid, nothing finished is lost, and `start` still resumes correctly. `stop` is just the
tidy way to do it.

## About the external drive (D:)

Only step 2 reads from D:. Keep D: connected until the `status` line says step 2 is
finished (it reprocesses the 8 days, about 6 hours of work, and you can pause it as often
as you like). After step 2, you can unplug D: and run everything else anywhere.

If D: disconnects during step 2, the script stops cleanly and writes how to recover in
`STATUS.txt`. Usually: reconnect the drive. If `ls /mnt/d` says "No such device", open
PowerShell and run `wsl --shutdown`, then reopen WSL and run `start` again.

## Roughly how long

About 39 hours of laptop time in total, spread over as many short sittings as you like.
The heaviest single step is the federated ablation (step 10, about 8 hours). Step 2
(about 6 hours) is the only one that needs D:.

## When it finishes

The last step writes `results_revision2.zip` into `~/rev2_work` and, if D: is connected,
copies it to `D:\fl-ids-sweep\`. Send me that zip. If D: is not connected, the file is at
`~/rev2_work/results_revision2.zip`; from Windows you can reach it at
`\\wsl$\Ubuntu\home\sangeetha\rev2_work\results_revision2.zip`.

## If something looks stuck

- `status` says the same thing for a long time: a federated run takes about 40 minutes,
  so that is normal. Step 2 days take a while too.
- You want to see live detail: `tail -f ~/rev2_work/STATUS.txt`
- You started it twice by mistake: it refuses to run a second copy, so that is safe.
- To see what it would do without running anything: `bash ~/rev2_work/rev2/run_revision2.sh dry-run`
