import os
import subprocess
import math
import inspect
from inspect import currentframe, getframeinfo

content = list()
linenum = list()

def register_line(code: str) -> None:
    lnum = inspect.stack()[-1].lineno

    for lidx, line in enumerate(code.split('\n')):
        content.append(line)
        linenum.append(lnum + lidx)

# Header
register_line(code='''module BCShifter128 #(  // Bubble-Collapsing Shifter
    parameter WORD_WIDTH    = 8,
    parameter PSUM_WIDTH    = 8,
    parameter DIST_WIDTH    = 7,
    parameter MAX_LIFM_RSIZ = 4
) (
    input [1023:0] psum,
    input [127:0]  mask,

    input [128*WORD_WIDTH-1:0]               lifm_line,
    input [128*DIST_WIDTH*MAX_LIFM_RSIZ-1:0] mt_line,

    output [128*WORD_WIDTH-1:0]               lifm_comp,
    output [128*DIST_WIDTH*MAX_LIFM_RSIZ-1:0] mt_comp
);

genvar line_idx;  // line index iterator

// Generate array connected with input and output ports
wire [WORD_WIDTH-1:0]               lifm_line_arr [0:127];
wire [DIST_WIDTH*MAX_LIFM_RSIZ-1:0] mt_line_arr   [0:127];

reg [WORD_WIDTH-1:0]               lifm_comp_arr [0:127];
reg [DIST_WIDTH*MAX_LIFM_RSIZ-1:0] mt_comp_arr   [0:127];

generate
    for (line_idx = 0; line_idx < 128; line_idx = line_idx+1) begin
        assign lifm_line_arr[line_idx] = lifm_line[WORD_WIDTH*line_idx+:WORD_WIDTH];
        assign mt_line_arr[line_idx] = mt_line[DIST_WIDTH*MAX_LIFM_RSIZ*line_idx+:DIST_WIDTH*MAX_LIFM_RSIZ];
        assign lifm_comp[WORD_WIDTH*line_idx+:WORD_WIDTH] = lifm_comp_arr[line_idx];
        assign mt_comp[DIST_WIDTH*MAX_LIFM_RSIZ*line_idx+:DIST_WIDTH*MAX_LIFM_RSIZ] = mt_comp_arr[line_idx];
    end
endgenerate
''')

# Generate shifters
for numel in range(2, 129, 1):
    register_line(code=f"""
// Shifter {numel}
wire [{numel}*WORD_WIDTH-1:0] i_lifm_l{numel};
wire [{numel}*WORD_WIDTH-1:0] o_lifm_l{numel};
wire [{numel}*DIST_WIDTH*MAX_LIFM_RSIZ-1:0] i_mt_l{numel};
wire [{numel}*DIST_WIDTH*MAX_LIFM_RSIZ-1:0] o_mt_l{numel};
wire [{math.ceil(math.log2(numel+1))-1}:0] stride_l{numel};

assign i_lifm_l{numel} = {{lifm_line_arr[{numel-1}], {{{numel-1}*WORD_WIDTH{{1'b0}}}}}};
assign i_mt_l{numel} = {{mt_line_arr[{numel-1}], {{{numel-1}*DIST_WIDTH*MAX_LIFM_RSIZ{{1'b0}}}}}};
assign stride_l{numel} = psum[{numel-1}*PSUM_WIDTH+:{math.ceil(math.log2(numel+1))}];

VShifter #(
    .WORD_WIDTH(WORD_WIDTH), .NUMEL({numel}), .NUMEL_LOG({math.ceil(math.log2(numel+1))})
) vs_lifm_{numel} (
    .i_vec(i_lifm_l{numel}), .stride(stride_l{numel}), .o_vec(o_lifm_l{numel})
);

VShifter #(
    .WORD_WIDTH(DIST_WIDTH*MAX_LIFM_RSIZ), .NUMEL({numel}), .NUMEL_LOG({math.ceil(math.log2(numel+1))})
) vs_mt_{numel} (
    .i_vec(i_mt_l{numel}), .stride(stride_l{numel}), .o_vec(o_mt_l{numel})
);""")

# Generate output signal
for idx in range(128):
    register_line(code=f"""
assign lifm_comp[{idx}] = {' | '.join([f'o_lifm_l{numel}[{idx}*WORD_WIDTH+:WORD_WIDTH]' for numel in range(max(idx+1, 2), 129, 1)])};
assign mt_comp[{idx}]   = {' | '.join([f'o_mt_l{numel}[{idx}*DIST_WIDTH*MAX_LIFM_RSIZ+:DIST_WIDTH*MAX_LIFM_RSIZ]' for numel in range(max(idx+1, 2), 129, 1)])};""")

# Tail
register_line(code='''
endmodule


module VShifter #(
    parameter WORD_WIDTH = 8,
    parameter NUMEL      = 128,
    parameter NUMEL_LOG  = 7
) (
    input [WORD_WIDTH*NUMEL-1:0] i_vec,
    input [NUMEL_LOG-1:0]        stride,

    output [WORD_WIDTH*NUMEL-1:0] o_vec
);

assign o_vec = i_vec >> (stride * WORD_WIDTH);
    
endmodule''')

# Save source code as Verilog file (.v)
dirname = os.path.join(os.curdir, 'code_gen')
filename = 'bubble_collapse_shifter'
vfilename = filename + '.v'
ofilename = filename + '.vvp'
clog_filename = filename + '_compile.log'

os.makedirs(dirname, exist_ok=True)

# Compile and check error
with open(os.path.join(dirname, vfilename), 'wt') as file:
    file.write('\n'.join(content))

with open(os.path.join(dirname, clog_filename), 'wt') as file:
    compile_result = subprocess.run(f"iverilog -o \"{os.path.join(dirname, ofilename)}\" {os.path.join(dirname, vfilename)}", stdout=file, stderr=file)

elinenum = []
enames = {}

with open(os.path.join(dirname, clog_filename), 'rt') as file:
    for eline in file.readlines():
        eparsed = eline.split(':')

        if len(eparsed) < 3:
            continue

        efile = eparsed[0]
        eidx = int(eparsed[1]) - 1
        ename = ':'.join(eparsed[2:]).strip()

        if linenum[eidx] not in elinenum:
            elinenum.append(linenum[eidx])

        if linenum[eidx] not in enames.keys():
            enames[linenum[eidx]] = []

        if ename not in enames[linenum[eidx]]:
            enames[linenum[eidx]].append(ename)

if __name__ == '__main__':
    print(f"Compile Configs")
    print(f"- verilog file:      {os.path.join(dirname, vfilename)}")
    print(f"- output file:       {os.path.join(dirname, ofilename)}")
    print(f"- compile log file:  {os.path.join(dirname, clog_filename)}\n")

    for el in elinenum:
        enn = '\n'.join(f'  [{ei + 1}] {en}' for ei, en in enumerate(enames[el]))
        print(f"ln {el:4d}\n{enn}", end='\n')

    # for eoffset in range(0, len(linenum), 20):
    #     print(linenum[eoffset:eoffset+20])