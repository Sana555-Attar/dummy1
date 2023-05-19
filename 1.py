import openml
from sklearn import neighbors


openml.config.start_using_configuration_for_example()

task = openml.tasks.get_task(403)
data = openml.datasets.get_dataset(task.dataset_id)
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
run = openml.runs.run_model_on_task(clf, task, avoid_duplicate_runs=False)
# Publish the experiment on OpenML (optional, requires an API key).
# For this tutorial, our configuration publishes to the test server
# as to not crowd the main server with runs created by examples.
myrun = run.publish()
print(f"kNN on {data.name}: {myrun.openml_url}")