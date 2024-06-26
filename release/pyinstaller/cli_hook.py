import sys

if __name__ == "__main__":
    import multiprocessing

    from alphadia import cli

    # catch pyinstaller bug
    # subprocess.CalledProcessError: Command '['/Applications/alphaDIA.app/Contents/Frameworks/alphadia', '-sS', '-c', 'import platform; print(platform.mac_ver()[0])']' returned non-zero exit status 2.
    # [36481] Failed to execute script 'cli_hook' due to unhandled exception!
    if len(sys.argv) >= 3 and (sys.argv[1] == "-sS") and (sys.argv[2] == "-c"):
        import platform

        print(platform.mac_ver()[0])
        sys.exit(0)

    multiprocessing.freeze_support()
    cli.run()
