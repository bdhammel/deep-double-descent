from subprocess import Popen, PIPE


def run_command(command, file_path, flag):
    process = Popen([command, file_path, flag], stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()


def model_capacity():
    widths = range(120, 200, 5)
    for width in widths:
        print("Training network with width: ", width)
        run_command('python', './train.py', str(width))


if __name__ == '__main__':
    model_capacity()
