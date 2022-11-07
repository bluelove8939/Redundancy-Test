import os
import subprocess
import inspect


class VerilogGenerator(object):
    def __init__(self, dirname: str=os.path.join(os.curdir, '..', 'code_gen'), filename: str='module'):
        super(VerilogGenerator, self).__init__()

        self.dirname = dirname
        self.filename = filename
        self.vfilename = filename + '.v'
        self.ofilename = filename + '.vvp'
        self.clog_filename = filename + '_compile.log'

        self.content = list()
        self.linenum = list()

        self.elinenum = list()
        self.enames = dict()

        os.makedirs(dirname, exist_ok=True)

    def reset(self):
        self.content = list()
        self.linenum = list()

    def register_line(self, code: str) -> None:
        lnum = inspect.stack()[-1].lineno

        for lidx, line in enumerate(code.split('\n')):
            self.content.append(line)
            self.linenum.append(lnum + lidx)

    def compile(self):
        with open(os.path.join(self.dirname, self.vfilename), 'wt') as file:
            file.write('\n'.join(self.content))

        with open(os.path.join(self.dirname, self.clog_filename), 'wt') as file:
            compile_result = subprocess.run(
                f"iverilog -o \"{os.path.join(self.dirname, self.ofilename)}\" {os.path.join(self.dirname, self.vfilename)}", stdout=file,
                stderr=file)

        self.elinenum = []
        self.enames = {}

        with open(os.path.join(self.dirname, self.clog_filename), 'rt') as file:
            for eline in file.readlines():
                eparsed = eline.split(':')

                if len(eparsed) < 3:
                    continue

                efile = eparsed[0]
                eidx = int(eparsed[1]) - 1
                ename = ':'.join(eparsed[2:]).strip()

                if self.linenum[eidx] not in self.elinenum:
                    self.elinenum.append(self.linenum[eidx])

                if self.linenum[eidx] not in self.enames.keys():
                    self.enames[self.linenum[eidx]] = []

                if ename not in self.enames[self.linenum[eidx]]:
                    self.enames[self.linenum[eidx]].append(f"{ename}  {efile}")

    def print_result(self):
        print(f"Compile Configs")
        print(f"- verilog file:      {os.path.join(self.dirname, self.vfilename)}")
        print(f"- output file:       {os.path.join(self.dirname, self.ofilename)}")
        print(f"- compile log file:  {os.path.join(self.dirname, self.clog_filename)}\n")

        for el in self.elinenum:
            enn = '\n'.join(f'  [{ei + 1}] {en}' for ei, en in enumerate(self.enames[el]))
            print(f"ln {el:4d}\n{enn}", end='\n')