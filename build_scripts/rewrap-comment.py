import itertools
import os
import operator
import sys
import textwrap

directory = sys.argv[1]

for path, subdirs, files in os.walk(directory):
    for filename in files:
        path_file = os.path.abspath(os.join.path(path, filename))
        formatted_code = []

        with open(path_file, "r") as f:
            content = f.read().split('\n')
            line_number_comment = [
                i for i, line in enumerate(content) if line.startswith('# ')
            ]
            grouped_line_number = []
            for k, g in itertools.groupby(enumerate(line_number_comment),
                                        key=lambda x: x[0] - x[1]):
                grouped_line_number.append(list(map(operator.itemgetter(1), g)))
            for gln_idx, gln in enumerate(grouped_line_number):
                # do not format the python code
                if gln_idx > 0:
                    start = grouped_line_number[gln_idx-1][-1] + 1
                    end = gln[0]
                    code_block = content[start:end]
                    code_block = "\n".join(code_block)
                    formatted_code.append('\n' + code_block + '\n')
                # do not format the jupytext metadata
                if content[gln[0]] == '# ---' and content[gln[-1]] == '# ---':
                    formatted_code.append(
                        "\n".join(content[gln[0]:gln[-1] + 1]) + '\n'
                    )
                else:
                    # specific handling for itemize
                    item_chars = ('# - ', '# * ', '# +')

                    # do not modify these lines
                    cell_chars = ['# %']
                    header_chars = ['# #', '# ===', '# ---', '# ~~~']
                    link_chars = ['# [', '# !']
                    table_chars = ['# |']
                    html_chars = ['# <']
                    untouched_chars = tuple(cell_chars + header_chars + link_chars +
                                            table_chars + html_chars)

                    comment_block = []
                    line_discarded = []
                    for ln_idx in range(len(gln)):
                        if ln_idx in line_discarded:
                            continue
                        line = content[gln[ln_idx]]
                        block = []
                        if line.startswith(untouched_chars):
                            block = line
                        elif line.startswith(item_chars):
                            block.append(line)
                            for nln_idx in range(ln_idx + 1, len(gln)):
                                next_line = content[gln[nln_idx]]
                                if next_line.startswith('#   '):
                                    line_discarded.append(nln_idx)
                                    block.append(next_line[3:])
                                else:
                                    break
                            block = "".join(block)
                            block = textwrap.wrap(block, width=75)
                            block = "\n#   ".join(block)
                        else:
                            block.append(line)
                            for nln_idx in range(ln_idx + 1, len(gln)):
                                next_line = content[gln[nln_idx]]
                                if not next_line.startswith(untouched_chars +
                                                            item_chars):
                                    line_discarded.append(nln_idx)
                                    block.append(next_line[1:])
                                else:
                                    break
                            block = "".join(block)
                            block = textwrap.wrap(block, width=77)
                            block = "\n# ".join(block)

                        comment_block.append(block)

                    if comment_block:
                        formatted_code.append("\n".join(comment_block))

        full_file = "".join(formatted_code)

        with open(path_file, 'w') as f:
            f.write(full_file)
