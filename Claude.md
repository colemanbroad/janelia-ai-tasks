In python we shouldn't return None or throw Exceptions to handle errors or missing data, just use assertions with a good error string.

Configs should be loaded into a SimpleNamespace and passed around as argument to functions that would otherwise have many params.
Configs are data. They shouldn't contain complex stateful objects.
If I add a comment to a block of code, leave it unless things totally change.

