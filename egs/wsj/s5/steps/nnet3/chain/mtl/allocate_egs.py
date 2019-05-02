#!/usr/bin/env python3

# Copyright      2019 Desh Raj
# Apache 2.0.

""" This script generates examples for multitask training of neural network.
    This scripts produces 2 sets of files --
    egs.*.scp, egs.output.*.ark

    egs.*.scp are the SCP files of the training examples.
    egs.output.*.ark map from the key of the example to the name of
    the output-node in the neural net for that specific task, e.g.
    'output','xvec.output'.

    --egs-prefix option can be used to generate train and diagnostics egs files.
    If --egs-prefix=train_diagnostics. is passed, then the files produced by the
    script will be named with the prefix as "train_diagnostics."
    instead of "egs."
    i.e. the files produced are -- train_diagnostics.*.scp,
    train_diagnostics.output.*.ark, train_diagnostics.weight.*.ark and
    train_diagnostics.ranges.*.txt.
    The other egs-prefix options used in the recipes are "valid_diagnositics."
    for validation examples and "combine." for examples used for model
    combination.

    For chain training egs, the --egs-prefix option should be "cegs."

    You can call this script as (e.g.):

    allocate_multilingual_examples.py [opts] example-scp-lists
        multilingual-egs-dir

    allocate_multilingual_examples.py --block-size 512
        --lang2weight  "0.2,0.8" exp/lang1/egs.scp exp/lang2/egs.scp
        exp/multi/egs

"""

import os, argparse, sys, random
import logging
import traceback

sys.path.insert(0, 'steps')

logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Start generating MTL examples')


def get_args():

    parser = argparse.ArgumentParser(
        description=""" This script generates examples for MTL training
        of neural network by producing 2 sets of primary files
        as egs.*.scp, egs.output.*.ark.
        egs.*.scp are the SCP files of the training examples.
        egs.output.*.ark map from the key of the example to the name of
        the output-node in the neural net for that specific task, e.g.
        'output-2'.""",
        epilog="Called by steps/nnet3/chain/mtl/combine_egs.sh")

    parser.add_argument("--num-archives", type=int, default=None,
                        help="Number of archives to split the data into. (Note: in reality they are not "
                        "archives, only scp files, but we use this notation by analogy with the "
                        "conventional egs-creating script).")
    parser.add_argument("--block-size", type=int, default=512,
                        help="This relates to locality of disk access. 'block-size' is"
                        "the average number of examples that are read consecutively"
                        "from each input scp file (and are written in the same order to the output scp files)"
                        "Smaller values lead to more random disk access (during "
                        "the nnet3 training process).")
    parser.add_argument("--egs-prefix", type=str, default="egs.",
                        help="This option can be used to add a prefix to the filenames "
                        "of the output files. For e.g. "
                        "if --egs-prefix=combine. , then the files produced "
                        "by this script will be "
                        "combine.output.*.ark, combine.weight.*.ark, and combine.*.scp")
# now the positional arguments
    parser.add_argument("egs_scp_lists", nargs='+',
                        help="List of egs.scp files per input task."
                           "e.g. exp/chain/tdnn_1a/egs_dir/egs.scp exp/chain/tdnn_1a/egs_dir_xvec/egs.scp")
    parser.add_argument("egs_dir",
                        help="Name of output egs directory e.g. exp/chain/tdnn_1a/egs_dir_final")


    print(sys.argv, file=sys.stderr)
    args = parser.parse_args()

    return args


def read_lines(file_handle, num_lines):
    n_read = 0
    lines = []
    while n_read < num_lines:
        line = file_handle.readline()
        if not line:
            break
        lines.append(line.strip())
        n_read += 1
    return lines

def get_num_examples(scp):
    with open(scp) as fh:
        total = sum([1 for line in fh])
    return (total)

def process_mtl_egs(args):

    scp_asr, scp_xvec = args.egs_scp_lists.split()
   
    num_egs_asr = get_num_examples(scp_asr)
    num_egs_xvec = get_num_examples(scp_xvec)

    if not os.path.exists(os.path.join(args.egs_dir, 'info')):
        os.makedirs(os.path.join(args.egs_dir, 'info'))

    # Total number of egs in all languages
    tot_num_egs = num_egs_asr + num_egs_xvec
    num_archives = args.num_archives

    with open("{0}/info/{1}num_archives".format(args.egs_dir, args.egs_prefix), "w") as fh:
        print("{0}".format(num_archives), file=fh)

    logger.info("There are a total of {} examples in the input scp "
                "files.".format(tot_num_egs))
    logger.info("Number of blocks in each output archive will be approximately "
                "{}, and block-size is {}.".format(int(round(tot_num_egs / num_archives / args.block_size)),
                                                   args.block_size))
    
    num_remaining_egs = tot_num_egs
    for archive_index in range(num_archives + 1):  #  +1 is because we write to the last archive in two rounds
        num_remaining_archives = num_archives - archive_index
        num_remaining_blocks = float(num_remaining_egs) / args.block_size

        last_round = (archive_index == num_archives)
        if not last_round:
            num_blocks_this_archive = int(round(float(num_remaining_blocks) / num_remaining_archives))
            logger.info("Generating archive {} containing {} blocks...".format(archive_index, num_blocks_this_archive))
        else:  # This is the second round for the last archive. Flush all the remaining egs...
            archive_index = num_archives - 1
            num_blocks_this_archive = 2
            logger.info("Writing all the {} remaining egs to the last archive...".format(num_remaining_egs))

        out_scp_file_handle = open('{0}/{1}{2}.scp'.format(args.egs_dir, args.egs_prefix, archive_index + 1),
                                   'a' if last_round else 'w')
        eg_to_output_file_handle = open("{0}/{1}output.{2}.ark".format(args.egs_dir, args.egs_prefix, archive_index + 1),
                                        'a' if last_round else 'w')

        # Write egs from both tasks alternatively
        for block_index in range(num_blocks_this_archive):
            if (block_index % 2 == 0):
                # Read 'block_size' examples from the ASR scp and write them to the current output scp file:
                example_lines  = read_lines(asr_scp, args.block_size)
                for eg_line in example_lines:
                    eg_id = eg_line.split()[0]
                    print(eg_line, file=out_scp_file_handle)
                    print("{0} output output-xent".format(eg_id), file=eg_to_output_file_handle)
            
            else:
                # Read 'block_size' examples from the Xvector scp and write to current output scp file
                example_lines  = read_lines(xvec_scp, args.block_size)
                for eg_line in example_lines:
                    eg_id = eg_line.split()[0]
                    print(eg_line, file=out_scp_file_handle)
                    print("{0} xvec.output".format(eg_id), file=eg_to_output_file_handle)

            num_remaining_egs -= len(example_lines)

        out_scp_file_handle.close()
        eg_to_output_file_handle.close()

    logger.info("Finished generating {0}*.scp, {0}output.*.ark files"
                "Wrote a total of {1} examples to {2} archives"
                .format(args.egs_prefix, tot_num_egs - num_remaining_egs, num_archives))


def main():
    try:
        args = get_args()
        process_mtl_egs(args)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
