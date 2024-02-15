if __name__ == "__main__":
    from alphadia import cli
    import multiprocessing

    multiprocessing.freeze_support()
    cli.run()
