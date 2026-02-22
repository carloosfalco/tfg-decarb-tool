ValueError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/tfg-decarb-tool/app.py", line 903, in <module>
    fig1 = px.scatter(
        chart_df,
    ...<5 lines>...
        title="CAPEX vs CO₂ reduction (bubble size = annual benefit)",
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/plotly/express/_chart_types.py", line 69, in scatter
    return make_figure(args=locals(), constructor=go.Scatter)
File "/home/adminuser/venv/lib/python3.13/site-packages/plotly/express/_core.py", line 2705, in make_figure
    trace.update(patch)
    ~~~~~~~~~~~~^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/plotly/basedatatypes.py", line 5197, in update
    BaseFigure._perform_update(self, dict1, overwrite=overwrite)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/plotly/basedatatypes.py", line 3971, in _perform_update
    BaseFigure._perform_update(plotly_obj[key], val)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/plotly/basedatatypes.py", line 3992, in _perform_update
    plotly_obj[key] = val
    ~~~~~~~~~~^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/plotly/basedatatypes.py", line 4932, in __setitem__
    self._set_prop(prop, value)
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/plotly/basedatatypes.py", line 5276, in _set_prop
    raise err
File "/home/adminuser/venv/lib/python3.13/site-packages/plotly/basedatatypes.py", line 5271, in _set_prop
    val = validator.validate_coerce(val)
File "/home/adminuser/venv/lib/python3.13/site-packages/_plotly_utils/basevalidators.py", line 796, in validate_coerce
    self.raise_invalid_elements(some_invalid_els)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/_plotly_utils/basevalidators.py", line 328, in raise_invalid_elements
                raise ValueError(
    ...<10 lines>...
                )
