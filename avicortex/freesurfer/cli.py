"""Freesurfer's aparcstats2table script."""

# Original Version - Douglas Greve, MGH
# Rewrite - Krish Subramaniam, MGH
# $Id: aparcstats2table,v 1.24 2013/01/31 19:22:45 greve Exp $
from __future__ import annotations

import csv
import logging
import optparse
import os
import sys
import warnings
from typing import Any

from avicortex.freesurfer.exceptions import BadFileError
from avicortex.freesurfer.parsers import AparcStatsParser
from avicortex.freesurfer.types import Ddict
from avicortex.freesurfer.utils import sanitize_table
from avicortex.freesurfer.writer import TableWriter

# map of delimiter choices and string literals
delimiter2char = {"comma": ",", "tab": "\t", "space": " ", "semicolon": ";"}

warnings.filterwarnings("ignore", ".*negative int.*")

# globals
ch = logging.StreamHandler()
aparclogger = logging.getLogger("aparcstats2table")
aparclogger.setLevel(logging.INFO)
aparclogger.addHandler(ch)


HELPTEXT = """
Converts a cortical stats file created by recon-all and or
mris_anatomical_stats (eg, ?h.aparc.stats) into a table in which
each line is a subject and each column is a parcellation. By
default, the values are the area of the parcellation in mm2. The
first row is a list of the parcellation names. The first column is
the subject name. If the measure is thickness then the last column
is the mean cortical thickness.

The subjects list can be specified on either of two ways:
  1. Specify each subject after a -s flag

            -s subject1 -s subject2 ... --hemi lh

  2. Specify all subjects after --subjects flag. --subjects does not have
     to be the last argument. Eg:

            --subjects subject1 subject2 ... --hemi lh

By default, it looks for the ?h.aparc.stats file based on the
Killiany/Desikan parcellation atlas. This can be changed with
'--parc parcellation' where parcellation is the parcellation to
use. An alternative is aparc.a2009s which was developed by
Christophe Destrieux. If this file is not found, it will exit
with an error unless --skip in which case it skips this subject
and moves on to the next.

By default, the area (mm2) of each parcellation is reported. This can
be changed with '--meas measure', where measure can be area, volume
(ie, volume of gray matter), thickness, thicknessstd, or meancurv.
thicknessstd is the standard dev of thickness across space.

Example:
 aparcstats2table --hemi lh --subjects 004 008 --parc aparc.a2009s
    --meas meancurv -t lh.a2009s.meancurv.txt

lh.a2009s.meancurv.txt will have 3 rows: (1) 'header' with the name
of each structure, (2) mean curvature for each structure for subject

The --common-parcs flag writes only the ROIs which are common to all
the subjects. Default behavior is it puts 0.0 in the measure of an ROI
which is not present in a subject.

The --parcs-from-file <file> outputs only the parcs specified in the file
The order of the parcs in the file is maintained. Specify one parcellation
per line.

The --report-rois flag, for each subject, gives what ROIs that are present
in at least one other subject is absent in current subject and also gives
what ROIs are unique to the current subject.

The --transpose flag writes the transpose of the table.
This might be a useful way to see the table when the number of subjects is
relatively less than the number of ROIs.

The --delimiter option controls what character comes between the measures
in the table. Valid options are 'tab' ( default), 'space', 'comma' and
'semicolon'.

The --skip option skips if it can't find a .stats file. Default behavior is
exit the program.

The --parcid-only flag writes only the ROIs name in the 1st row 1st column
of the table. Default is hemi_ROI_measure
"""


def callback_var(option: Any, opt_str: Any, value: Any, parser: Any) -> None:
    """Allow callback to handle variable number of arguments for an option instead of optparse."""
    value = []
    rargs = parser.rargs
    idx_arg = 2
    while rargs:
        arg = rargs[0]
        if (arg[:idx_arg] == "--" and len(arg) > idx_arg) or (
            arg[:1] == "-" and len(arg) > 1 and arg[1] != "-"
        ):
            break
        else:
            value.append(arg)
            del rargs[0]
    setattr(parser.values, option.dest, value)


def check_subjdirs() -> str:
    """
    Quit if SUBJECTS_DIR is not defined as an environment variable.

    This is not a function which returns a boolean. Execution is stopped if not found.
    If found, returns the SUBJECTS_DIR
    """
    if "SUBJECTS_DIR" not in os.environ:
        logging.info("ERROR: SUBJECTS_DIR environment variable not defined!")
        sys.exit(1)
    return os.environ["SUBJECTS_DIR"]


def sanity_check(options: Any) -> None:
    """Check if inputs make sense."""
    if options.subjects is not None and len(options.subjects) < 1:
        logging.info("ERROR: at least 1 subject must be provided")
        sys.exit(1)

    if (
        options.subjects is None
        and options.subjectsfile is None
        and options.qdec is None
        and options.qdeclong is None
    ):
        logging.info(
            "ERROR: Specify one of --subjects, --subjectsfile --qdec or --qdec-long"
        )
        logging.info("       or run with --help for help.")
        sys.exit(1)

    count = 0
    if options.subjects is not None:
        count = count + 1
    if options.subjectsfile is not None:
        count = count + 1
    if options.qdec is not None:
        count = count + 1
    if options.qdeclong is not None:
        count = count + 1
    if count > 1:
        logging.info(
            "ERROR: Please specify just one of  --subjects, --subjectsfile --qdec or --qdec-long."
        )
        sys.exit(1)

    if not options.outputfile:
        logging.info("ERROR: output table name should be specified")
        sys.exit(1)
    if not options.hemi:
        logging.info("ERROR: hemisphere should be provided (lh or rh)")
        sys.exit(1)

    # parse the parcs file
    options.parcs = None
    if options.parcsfile is not None:
        try:
            with open(options.parcsfile, encoding="utf-8") as fp:
                options.parcs = [line.strip() for line in fp]
        except OSError:
            logging.info("ERROR: cannot read " + options.parcsfile)


def options_parse() -> Any:  # noqa: PLR0914
    """
    Command Line Options Parser for aparcstats2table.

    initiate the option parser and return the parsed object
    """
    parser = optparse.OptionParser(
        version="$Id: aparcstats2table,v 1.24 2013/01/31 19:22:45 greve Exp $",
        usage=HELPTEXT,
    )

    # help text
    h_sub = "(REQUIRED) subject1 <subject2 subject3..>"
    h_s = " subjectname"
    h_subf = "name of the file which has the list of subjects ( one subject per line)"
    h_qdec = "name of the qdec table which has the column of subjects ids (fsid)"
    h_qdeclong = "name of the longitudinal qdec table with column of tp ids (fsid) and subject templates (fsid-base)"
    h_hemi = "(REQUIRED) lh or rh"
    h_parc = "parcellation.. default is aparc ( alt aparc.a2009s)"
    h_meas = "measure: default is area ( alt volume, thickness, thicknessstd, meancurv, gauscurv, foldind, curvind)"
    h_skip = "if a subject does not have input, skip it"
    h_t = "(REQUIRED) output table file"
    h_deli = "delimiter between measures in the table. default is tab (alt comma, space, semicolon )"
    h_parcid = "do not pre/append hemi/meas to parcellation name"
    h_common = "output only the common parcellations of all the subjects given"
    h_parcfile = "filename: output parcellations specified in the file"
    h_roi = "logging.info ROIs information for each subject"
    h_tr = "transpose the table ( default is subjects in rows and ROIs in cols)"
    h_v = "increase verbosity"

    # Add the options
    parser.add_option(
        "--subjects",
        dest="subjects",
        action="callback",
        callback=callback_var,
        help=h_sub,
    )
    parser.add_option("-s", dest="subjects", action="append", help=h_s)
    parser.add_option("--subjectsfile", dest="subjectsfile", help=h_subf)
    parser.add_option("--qdec", dest="qdec", help=h_qdec)
    parser.add_option("--qdec-long", dest="qdeclong", help=h_qdeclong)
    parser.add_option("--hemi", dest="hemi", choices=("lh", "rh"), help=h_hemi)
    parser.add_option("-t", "--tablefile", dest="outputfile", help=h_t)
    parser.add_option("-p", "--parc", dest="parc", default="aparc", help=h_parc)
    parser.add_option(
        "-m",
        "--measure",
        dest="meas",
        choices=(
            "area",
            "volume",
            "thickness",
            "thicknessstd",
            "meancurv",
            "gauscurv",
            "foldind",
            "curvind",
        ),
        default="area",
        help=h_meas,
    )
    parser.add_option(
        "-d",
        "--delimiter",
        dest="delimiter",
        choices=("comma", "tab", "space", "semicolon"),
        default="tab",
        help=h_deli,
    )
    parser.add_option(
        "--skip", action="store_true", dest="skipflag", default=False, help=h_skip
    )
    parser.add_option(
        "--parcid-only",
        action="store_true",
        dest="parcidflag",
        default=False,
        help=h_parcid,
    )
    parser.add_option(
        "--common-parcs",
        action="store_true",
        dest="commonparcflag",
        default=False,
        help=h_common,
    )
    parser.add_option("--parcs-from-file", dest="parcsfile", help=h_parcfile)
    parser.add_option(
        "--report-rois",
        action="store_true",
        dest="reportroiflag",
        default=False,
        help=h_roi,
    )
    parser.add_option(
        "",
        "--transpose",
        action="store_true",
        dest="transposeflag",
        default=False,
        help=h_tr,
    )
    parser.add_option(
        "-v",
        "--debug",
        action="store_true",
        dest="verboseflag",
        default=False,
        help=h_v,
    )
    options, _ = parser.parse_args()

    sanity_check(options)

    if options.reportroiflag:
        logging.info("WARNING: --report-rois deprecated. Use -v instead")

    if options.verboseflag:
        aparclogger.setLevel(logging.DEBUG)

    return options


def assemble_inputs(options: Any) -> Any:  # noqa: C901
    """
    Organize inputs.

    Parameters
    ----------
        the parsed 'options'

    Returns
    -------
        a list of tuples of (specifier names ( subjects), path to the corresponding .stats files)
    """
    o = options
    specs_paths = []
    # check the subjects dir
    subjdir = check_subjdirs()
    logging.info(subjdir)
    # in case the user gave --subjectsfile argument
    if o.subjectsfile is not None:
        o.subjects = []
        try:
            with open(o.subjectsfile, encoding="utf-8") as fp:
                for subfromfile in fp:
                    o.subjects.append(subfromfile.strip())
        except OSError:
            logging.info(f"ERROR: the file {o.subjectsfile} does not exist")
            sys.exit(1)
    if o.qdec is not None:
        o.subjects = []
        try:
            with open(o.qdec, encoding="utf-8") as f:
                dialect = csv.Sniffer().sniff(f.read(1024))
                f.seek(0)
                reader = csv.DictReader(f, dialect=dialect)
                # o.subjects = [row['fsid'] for row in reader]
                for row in reader:
                    fsid = row["fsid"].strip()
                    if fsid[0] != "#":
                        o.subjects.append(fsid)
                logging.info(o.subjects)
        except OSError:
            logging.info(f"ERROR: the file {o.qdec} does not exist")
            sys.exit(1)
    if o.qdeclong is not None:
        o.subjects = []
        try:
            with open(o.qdeclong, encoding="utf-8") as f:
                dialect = csv.Sniffer().sniff(f.read(1024))
                f.seek(0)
                reader = csv.DictReader(f, dialect=dialect)
                # o.subjects = [(row['fsid']+'.long.'+row['fsid-base']) for row in reader]
                for row in reader:
                    fsid = row["fsid"].strip()
                    if fsid[0] != "#":
                        o.subjects.append(fsid + ".long." + row["fsid-base"].strip())
        except OSError:
            logging.info(f"ERROR: the file {o.qdeclong} does not exist")
            sys.exit(1)

    for sub in o.subjects:
        specs_paths.append(
            (sub, os.path.join(subjdir, sub, "stats", o.hemi + "." + o.parc + ".stats"))
        )
    return specs_paths


def write_table(options: Any, rows: list[Any], cols: Any, table: Ddict) -> None:
    """Write the table from memory to disk. Initialize the writer class."""
    tw = TableWriter(rows, cols, table)
    r1c1 = f"{options.hemi}.{options.parc}.{options.meas}"
    tw.assign_attributes(
        filename=options.outputfile,
        row1col1=r1c1,
        delimiter=delimiter2char[options.delimiter],
    )
    # we might need the hemisphere and measure info in columns as well
    if not options.parcidflag:
        tw.decorate_col_titles(options.hemi + "_", "_" + options.meas)
    if options.transposeflag:
        tw.write_transpose()
    else:
        tw.write()


def main() -> None:
    """Define main entrypoint for the CLI."""
    # Command Line options are error checking done here
    options = options_parse()
    aparclogger.debug("-- The options you entered --")
    aparclogger.debug(options)

    # Assemble the input stats files
    subj_listoftuples = assemble_inputs(options)

    # Init the table in memory
    # is a list containing tuples of the form
    # [(specifier, segidlist, structlist, measurelist),] for all specifiers
    pretable = []

    # Parse the parc.stats files
    logging.info("Parsing the .stats files")
    for specifier, filepath in subj_listoftuples:
        try:
            aparclogger.debug("-" * 20)
            aparclogger.debug("Processing file " + filepath)
            parsed = AparcStatsParser(filepath)
            # parcs filter from the command line
            if options.parcs is not None:
                parsed.parse_only(options.parcs)

            parc_measure_map = parsed.parse(options.meas)
            aparclogger.debug("-- Parsed Parcs and Measures --")
            aparclogger.debug(parc_measure_map)
        except BadFileError as e:
            if options.skipflag:
                logging.info("Skipping " + str(e))
                continue
            else:
                logging.info(
                    "ERROR: The stats file "
                    + str(e)
                    + " is not found or is too small to be a valid statsfile"
                )
                logging.info("Use --skip flag to automatically skip bad stats files")
                sys.exit(1)

        pretable.append((specifier, parc_measure_map))

    # Make sure the table has the same number of cols for all stats files
    # and merge them up, clean them up etc. More in the documentation of the fn.
    logging.info("Building the table..")
    rows, columns, table = sanitize_table(pretable, options.commonparcflag)

    # Write this table ( in memory ) to disk.. function uses TableWriter class
    logging.info(f"Writing the table to {options.outputfile}")
    write_table(options, rows, columns, table)

    # always exit with 0 exit code
    sys.exit(0)


if __name__ == "__main__":
    main()
