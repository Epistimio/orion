
import tempfile
import os
import subprocess
import traceback
import logging
from contextlib import contextmanager
import time
from multiprocessing import Manager, Process
import shutil


log = logging.getLogger(__file__)

def launch_mongod(shared, port, address, dir) -> None:
    """Execute the mongoDB server in a subprocess."""
    arguments = [
        '--dbpath', dir,
        '--wiredTigerCacheSizeGB', '1',
        '--port', str(port),
        '--bind_ip', address,
        '--pidfilepath', os.path.join(dir, 'pid')
    ]

    kwargs = dict(
        args=' '.join(['mongod'] + arguments),
        stdout=subprocess.PIPE,
        bufsize=1,
        stderr=subprocess.STDOUT,
    )

    shared['db_pid_path'] = os.path.join(dir, 'pid')

    with subprocess.Popen(**kwargs, shell=True) as proc:
        try:
            shared['pid'] = proc.pid

            if 'dp_pid' not in shared:
                try:
                    shared['dp_pid'] = int(open(os.path.join(dir, 'pid'), 'r').read())
                    shared['running'] = True
                except FileNotFoundError:
                    pass

            while shared['running']:
                if proc.poll() is None:
                    line = proc.stdout.readline().decode('utf-8')[:-1]
                    log.debug(line)
                else:
                    shared['running'] = False
                    shared['exit'] = proc.returncode
                    log.debug('Stopping mongod popen')

        except Exception:
            log.error(traceback.format_exc())
            shared['running'] = False
            return

@contextmanager
def mongod(port, address) -> None:
    """Launch a mongoDB server in parallel. The server is stop on exit"""

    with tempfile.TemporaryDirectory() as dir:
        with Manager() as manager:
            shared = manager.dict()
            shared['running'] = False

            proc = Process(target=launch_mongod, args=(shared, port, address, dir))
            proc.start()

            acc = 0
            while not shared['running'] and acc < 2:
                acc += 0.01
                time.sleep(0.01)

            log.debug('Mongod ready after %f', acc)

            yield proc

            log.debug("Should stop")

            shared['running'] = False
            subprocess.run(['mongod', '--shutdown', '--dbpath', dir])

            log.debug('Stopping mongod process')
            shutil.rmtree(dir)
            proc.kill()