PYTHON_SCRIPTS_DIR = python_scripts
NOTEBOOKS_DIR = notebooks
RENDERED_NOTEBOOKS_DIR = rendered_notebooks
JUPYTER_KERNEL := python3

$(NOTEBOOKS_DIR): $(NOTEBOOKS_DIR)/*.ipynb

$(RENDERED_NOTEBOOKS_DIR): $(RENDERED_NOTEBOOKS_DIR)/*.ipynb

$(NOTEBOOKS_DIR)/%.ipynb: $(PYTHON_SCRIPTS)/%.py
	jupytext --set-formats notebooks//ipynb,python_scripts//py:percent $<
	jupytext --sync $<

$(RENDERED_NOTEBOOKS_DIR)/%.ipynb: $(NOTEBOOKS_DIR)/%.ipynb
	cp $< $@;\
	jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=None \
                    --ExecutePreprocessor.kernel_name=$(JUPYTER_KERNEL) --inplace $@
