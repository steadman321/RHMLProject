To run the unit tests : 
- cd to the Project root dir
- To run all tests : 
    python -m unittest discover -s unit_tests
- To run a subset of tests matching a filename pattern e.g for all regression based tests:
    python -m unittest discover -s unit_tests -p "*Regression*.py"

-To run the unittests and get a readable report, use the test runner like this :
    python /Users/jsteadman/js/DEV/Project/unit_tests/testRunner.py


NOTE: these unti tests have canned-output type results, so may not work on other envs; you may need to update the test expected output to fit your env
