import os


def parse_csvfile(filepath):
    parsed = {}

    with open(filepath, 'rt') as file:
        content = file.readlines()
        comment_removed = list(filter(lambda x: not (x.startswith('#') or len(x.strip()) == 0), content))
        header = [h.strip() for h in comment_removed[0].split(',')]  # consider first row as a header

        for line in comment_removed[1:]:
            line_splitted = [el.strip() for el in line.split(',')]
            parsed[line_splitted[0]] = {h: float(el) for h, el in zip(header[1:], line_splitted[1:])}

    return parsed


if __name__ == '__main__':
    result_dirname = os.path.join(os.curdir, 'results', 'pruned_weight_redundancy')
    output_dirname = os.path.join(os.curdir, 'results', 'pruned_weight_redundancy_analysis_by_step_range')
    output_filename = 'ratios_result.csv'

    filepath = []    # target files
    raw_ratios = {}  # raw ratio data
    results = {}     # processed ratio data


    # Read all file paths
    for filename in os.listdir(result_dirname):
        if os.path.isfile(os.path.join(result_dirname, filename)):
            filepath.append(os.path.join(result_dirname, filename))

    # Read all files
    for fp in filepath:
        with open(fp, 'rt') as file:
            # Generaing raw results (number of redundant operations)
            raw_results = parse_csvfile(fp)

            # Calculate redundant operation ratios
            tsum, msum = 0, 0
            for k, v in raw_results.items():
                tsum += v['total']
                msum += v['matched']

            ratios = {}

            for lname, lresult in raw_results.items():
                # total = np.sum(np.array(list(lresult.values())))
                ratios[lname] = float(lresult['matched'] / (lresult['total'] + 1e-7))

            ratios['overall ratio'] = msum / tsum

            # Save calculated ratios
            _, filename = os.path.split(fp)
            model_name, step_range = filename.split('_')
            step_range = step_range.split('.')[0]
            step_range = int(step_range) if step_range != 'None' else -1

            if model_name not in raw_ratios.keys():
                raw_ratios[model_name] = {}

            raw_ratios[model_name][step_range] = ratios

    # Processing raw ratio data
    for model_name, mresults in raw_ratios.items():
        results[model_name] = {}

        for sr in sorted((mresults.keys())):
            for lname, rval in mresults[sr].items():
                if lname not in results[model_name].keys():
                    results[model_name][lname] = {}

                results[model_name][lname][sr] = rval

    print(results)

    # Save processed ratios as CSV file format
    os.makedirs(output_dirname, exist_ok=True)

    with open(os.path.join(output_dirname, output_filename), 'wt') as ofile:
        for model_name, mresults in results.items():
            ofile.write("# Model Name: " + model_name + '\n\n')
            is_header_added = False

            for lname, rresults in mresults.items():
                if not is_header_added:
                    ofile.write(f"{'layer name':30s}, " + ', '.join([f'{"step" + str(h) if h != -1 else "no step":>8s}' for h in sorted(rresults.keys())]) + '\n')
                    is_header_added = True

                ofile.write(f"{lname:30s}, " + ', '.join([f'{rresults[h]*100:8.2f}' for h in sorted(rresults.keys())]) + '\n')

            ofile.write('\n\n\n')

