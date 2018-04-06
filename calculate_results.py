from mutation_calc import *
from pitest_html_parser import *
import glob
import os


def calculate_results(default_dir='results/'):
    """Aggregates all the mutations calculated and add the information about the mutation

    Arguments
    -------------
    - default_dir: the dir where all the csv files are stores

    """
    if not os.path.isdir(default_dir):
        print("No dir to process! You're missing some previous steps")
        exit(0)
    result_csv = glob.glob(default_dir+'res_*.csv')
    frames = []
    for project in result_csv:
        frames.append(pd.read_csv(project))
    aggregate = pd.concat(frames)
    print("* Aggregating results for mutation score")
    aggregate['mutation'] = aggregate.apply(lambda x: get_mutation_value_html(x), axis=1)
    print("* Aggregating results for number of mutations")
    aggregate['no_mutations'] = aggregate.apply(lambda x: get_mutation_value_html(x, True), axis=1)
    print("* Aggregating results for line coverage")
    aggregate['line_coverage'] = aggregate.apply(lambda x: get_mutation_value_html(x, False, True), axis=1)
    print("* Saving overall results in result.csv")
    aggregate.to_csv('results.csv', index=False)


def get_mutation_value_html(row, ret_no_mutation=False, ret_line_cov=False):
    """Functions called into the lambda to calculate the mutation
    The output from pitest needs to be in HTML format

    Arguments
    -------------
    - row: the row of the of the data frame
    - ret_no_mutation: if true, returns the number of mutations; if false, the mutation score

    """
    path = 'mutation_results/' + row.project + '/' + row.test_name + '/**/index.html'
    mutation_file = glob.glob(path, recursive=True)
    if not mutation_file:
        # no file = no mutations
        return None
    html_parser = PitestHTMLParser(mutation_file[0])
    if ret_line_cov:
        return html_parser.get_line_coverage()
    if ret_no_mutation:
        return html_parser.get_no_mutants()
    return html_parser.get_mutation_coverage()


def get_mutation_value(row, ret_no_mutation=False):
    """Functions called into the lambda to calculate the mutation
    The output from pitest needs to be in CSV format

    Arguments
    -------------
    - row: the row of the of the data frame
    - ret_no_mutation: if true, returns the number of mutations; if false, the mutation score

    """
    path = 'mutation_results/' + row.project + '/' + row.test_name + '/**/mutations.csv'
    mutation_file = glob.glob(path, recursive=True)
    # used to reuse the expensive glob search
    if not mutation_file:
        return None
    if ret_no_mutation:
        return get_number_of_mutations(mutation_file[0])
    mutation = calc_coverage(mutation_file[0])
    return mutation


def get_number_of_mutations(path):
    """Returns the number of mutations for the class in the given path

    Arguments
    -------------
    - path: the path of the csv file for the current mutation

    """
    if not path:
        return 0
    return get_number_mutation(path)


if __name__ == '__main__':
    calculate_results()
