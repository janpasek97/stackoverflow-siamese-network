"""
Dataset export in textual format
"""
import csv
import sys
from data.documents import Post
from elasticsearch_dsl.connections import connections
from network.utils.text_cleaner import TextCleaner
from network.utils.code_cleaner import CodeCleaner
from data.documents import ES_HOSTS, ES_LOGIN


def export_dataset_as_text(ds_fn):
    """
    Reads in a csv dataset with format 'first_post_id, second_post_id, label' and
    produce two csv files:
        1) cleaned textual dataset with in csv with format 'first_post_text, second_post_text, label'
        2) cleaned code dataset with in csv with format 'first_post_code, second_post_code, label'
    The export contains tokens separated by ' '. No additional processing except .split(' ') is necessary.

    :param ds_fn: path to a csv file with dataset export in format 'first_post_id, second_post_id, label'
    :return: void
    """
    text_cleaner = TextCleaner()
    code_cleaner = CodeCleaner()
    connections.create_connection(hosts=ES_HOSTS, timeout=9999, http_auth=ES_LOGIN)

    # output files, one for code, second for text
    text_export_fn = ds_fn.replace(".csv", "") + "_text.csv"
    code_export_fn = ds_fn.replace(".csv", "") + "_code.csv"
    with open(ds_fn, "r") as original_ds:
        with open(text_export_fn, "w", encoding="utf-8") as text_export_ds:
            with open(code_export_fn, "w", encoding="utf-8") as code_export_ds:
                ds_reader = csv.reader(original_ds, delimiter=",")
                text_ds_writer = csv.writer(text_export_ds, delimiter=",", quoting=csv.QUOTE_MINIMAL)
                code_ds_writer = csv.writer(code_export_ds, delimiter=",", quoting=csv.QUOTE_MINIMAL)

                # iterate over the whole source csv file
                for row in ds_reader:
                    # parse the source row
                    first_post_id = row[0]
                    second_post_id = row[1]
                    label = row[2]

                    # collect the post bodies
                    first_post_content = Post.get(id=first_post_id).text
                    second_post_content = Post.get(id=second_post_id).text

                    # pre-process the text
                    first_post_text = text_cleaner.clean_text(first_post_content)
                    second_post_text = text_cleaner.clean_text(second_post_content)

                    # pre-process the code
                    first_post_code = code_cleaner.clean_code(first_post_content)
                    second_post_code = code_cleaner.clean_code(second_post_content)

                    # write output
                    text_ds_writer.writerow([first_post_text, second_post_text, label])
                    code_ds_writer.writerow([first_post_code, second_post_code, label])


if __name__ == '__main__':
    fn = sys.argv[1]
    print("Exporting dataset as text ...")
    export_dataset_as_text(fn)
    print("Finished")
