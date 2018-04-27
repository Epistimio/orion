
        self.space = experiment.space
        if self.space is None:
            raise RuntimeError("Experiment object provided to Consumer has not yet completed"
                               " initialization.")

        # Fetch space builder
        self.template_builder = SpaceBuilder()
        # Get path to user's script and infer trial configuration directory
        self.script_path = experiment.metadata['user_script']
        self.tmp_dir = os.path.join(tempfile.gettempdir(), 'orion')
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.converter = JSONConverter()

    def consume(self, trial):
        """Execute user's script as a block box using the options contained
        within `trial`.

        :type trial: `orion.core.worker.trial.Trial`

        """
        log.debug("### Create new temporary directory at '%s':", self.tmp_dir)
        with tempfile.TemporaryDirectory(prefix=self.experiment.name + '_',
                                         dir=self.tmp_dir) as workdirname:
            log.debug("## New temp consumer context: %s", workdirname)
            completed_trial = self._consume(trial, workdirname)

        if completed_trial is not None:
            log.debug("### Register successfully evaluated %s.", completed_trial)
            self.experiment.push_completed_trial(completed_trial)
        else:
            log.debug("### Save %s as broken.", trial)
            trial.status = 'broken'
            Database().write('trials', trial.to_dict(),
                             query={'_id': trial.id})

    def _consume(self, trial, workdirname):
        config_file = tempfile.NamedTemporaryFile(mode='w', prefix='trial_',
                                                  suffix='.conf', dir=workdirname,
                                                  delete=False)
        config_file.close()
        log.debug("## New temp config file: %s", config_file.name)
        results_file = tempfile.NamedTemporaryFile(mode='w', prefix='results_',
                                                   suffix='.log', dir=workdirname,
                                                   delete=False)
        results_file.close()
        log.debug("## New temp results file: %s", results_file.name)

        log.debug("## Building command line argument and configuration for trial.")
        cmd_args = self.template_builder.build_to(config_file.name, trial)

        log.debug("## Launch user's script as a subprocess and wait for finish.")
        script_process = self.launch_process(results_file.name, cmd_args)

        if script_process is None:
            return None

        returncode = script_process.wait()

        if returncode != 0:
            log.error("Something went wrong. Check logs. Process "
                      "returned with code %d !", returncode)
            return None

        log.debug("## Parse results from file and fill corresponding Trial object.")
        results = self.converter.parse(results_file.name)

        trial.results = [Trial.Result(name=res['name'],
                                      type=res['type'],
                                      value=res['value']) for res in results]

        return trial

    def launch_process(self, results_filename, cmd_args):
        """Facilitate launching a black-box trial."""
        env = dict(os.environ)
        env['ORION_RESULTS_PATH'] = str(results_filename)
        command = [self.script_path] + cmd_args
        process = subprocess.Popen(command, env=env)
        returncode = process.poll()
        if returncode is not None and returncode < 0:
            log.error("Failed to execute script to evaluate trial. Process "
                      "returned with code %d !", returncode)
            return None

        return process

