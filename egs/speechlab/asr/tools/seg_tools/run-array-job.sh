#!/bin/bash

# submit batch jobs

function usage() {
  echo "Usage: run-array-job.sh <list> <script> [arguments for script]"
  echo "Runs <script> with the arguments following the script name. Each"
  echo "instance of the argument \"SET\" in the arguments are replaced"
  echo "by the line in the file <list> specified by the number in the"
  echo "environment variable \$SGE_TASK_ID"
  echo "For example: Say the list file \"list\" contains the following"
  echo 
  echo apple
  echo banana
  echo cherry
  echo
  echo - and the variable \$SGE_TASK_ID is set to 2. Then running the
  echo following:
  echo
  echo run-array-job.sh list script foo SET bar
  echo 
  echo Will execute:
  echo
  echo script foo banana bar
  echo
  exit 1
}

# Check that there are at least the right number of arguments
if [ $# -lt 2 ]
then
    usage
fi

# get the command line args
list_file="$1"
script="$2"
shift
shift

# Is the list file readable?
if [ ! -r "$list_file" ]
then
    echo "Error: $list_file not found or no read permission."
    exit 1
fi


# Compute the value of $SET by finding the nth line
# in $list_file
# depends on the value of $SGE_TASK_ID being set
# in the calling shell.
set=`gawk '(NR==n)' n=$SGE_TASK_ID "$list_file"`

# Iterate over remaining arguments
i=1
while [ -n "$1" ]
do
  case "$1" in
      SET)
	  invocation[$i]="$set"
	  ;;
      *)
	  invocation[$i]="$1"
	  ;;
  esac
  shift
  i=$(($i + 1))
done

# Run the script
$script ${invocation[@]}
