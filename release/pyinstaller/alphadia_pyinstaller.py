if __name__ == "__main__":
    import alphadia.cli
    import multiprocessing

    multiprocessing.freeze_support()
    alphadia.cli.run()
