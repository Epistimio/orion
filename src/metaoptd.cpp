// LINUX sytem headers
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>

int main()
{
    std::string cmd = "python"; // secondary program you want to run

    pid_t pid = fork(); // create child process
    int status;

    switch (pid)
    {
    case -1: // error
        perror("fork");
        exit(1);

    case 0: // child process
        execl(cmd.c_str(), 0, 0); // run the command
        perror("execl"); // execl doesn't return unless there is a problem
        exit(1);

    default: // parent process, pid now contains the child pid
        while (-1 == waitpid(pid, &status, 0)); // wait for child to complete
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        {
            // handle error
            std::cerr << "process " << cmd << " (pid=" << pid << ") failed" << std::endl;
        }
        break;
    }
    return 0;
}

