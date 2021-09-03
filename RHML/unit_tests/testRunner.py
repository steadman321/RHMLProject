from unittest import TestLoader, TestSuite
from HtmlTestRunner import HTMLTestRunner
import testRegressionTrees
import testClassificationTrees
import testBoostrapper
import testBagger
import testBooster

regression_tree_tests       = TestLoader().loadTestsFromTestCase(testRegressionTrees.TestRegressionTrees)
classification_tree_tests   = TestLoader().loadTestsFromTestCase(testClassificationTrees.TestClassificationTrees)
boostrapper_tests           = TestLoader().loadTestsFromTestCase(testBoostrapper.TestBoostrapper)
bagger_tests                = TestLoader().loadTestsFromTestCase(testBagger.TestBagger)
booster_tests               = TestLoader().loadTestsFromTestCase(testBooster.TestBooster)

suite = TestSuite([ regression_tree_tests,
                    classification_tree_tests,
                    boostrapper_tests,
                    bagger_tests,
                    booster_tests
                    ])
# suite = TestSuite([ 
#                     booster_tests
#                     ])
runner = HTMLTestRunner(output='unittest_reports',combine_reports=True,report_name="RHMLEnsembleUnitTests")
runner.run(suite)