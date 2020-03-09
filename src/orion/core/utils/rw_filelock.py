# Extension of https://github.com/elarivie/pyReaderWriterLock for FileLock
# The MIT License (MIT)
# Copyright (c) 2019 Orion Authors
# Copyright (c) 2018 Éric Larivière

from filelock import FileLock, _Acquire_ReturnProxy


class RWFileLock():
    """A Read/Write lock using FileLock giving fairness to both Reader and Writer."""

    def __init__(self, basename, timeout=-1):
        self.read = FileLock(f'{basename}.rlock', timeout)
        self.write = FileLock(f'{basename}.wlock', timeout)
        self.read_count = FileLock(f'{basename}.rclock', timeout)

        self.read_count_file = f'{basename}.num_readers'
        with open(self.read_count_file, 'w') as f:
            f.write("00")


    def read_lock(self):
        return ReaderLock(self)

    def write_lock(self):
        return WriterLock(self)


class ReaderLock():
    def __init__(self, RWLock):
        self.lock = RWLock
        self.is_locked = False

    def acquire(self, timeout=None):
        with self.lock.read.acquire(timeout=timeout):
            with self.lock.read_count.acquire(timeout=timeout):
                with open(self.lock.read_count_file, 'r+') as f:
                    read_count = int(f.read())
                    if read_count == 0:
                        self.lock.write.acquire(timeout=timeout)

                    read_count += 1
                    f.write(f"{read_count:02}")
                    self.is_locked = True

        return _Acquire_ReturnProxy(lock=self)

    def release(self):
        with self.lock.read_count.acquire():
            with open(self.lock.read_count_file, 'r+') as f:
                try:
                    read_count = int(f.read())
                except ValueError:
                    raise Exception(f'read count lock file {self.lock.read_count.lock_file} has invalid number of readers')
                read_count -= 1

                if read_count == 0:
                    self.lock.write.release()

                f.write(f"{read_count:02}")
                self.is_locked = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        self.release()


class WriterLock():
    def __init__(self, RWLock):
        self.lock = RWLock
        self.is_locked = False

    def acquire(self, timeout=None):
        self.lock.read.acquire(timeout=timeout)
        self.lock.write.acquire(timeout=timeout)
        self.is_locked = True

        return _Acquire_ReturnProxy(lock=self)

    def release(self):
        self.is_locked = False
        self.lock.write.release()
        self.lock.read.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        self.release()

