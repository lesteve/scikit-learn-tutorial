PYTHON_SCRIPTS_DIR = python_scripts
NOTEBOOKS_DIR = notebooks
RENDERED_NOTEBOOKS_DIR = rendered_notebooks
JUPYTER_KERNEL := python3

.PHONY: $(NOTEBOOKS_DIR) $(RENDERED_NOTEBOOKS_DIR) sanity_check_$(PYTHON_SCRIPTS_DIR) sanity_check_$(NOTEBOOKS_DIR) sanity_check_$(RENDERED_NOTEBOOKS_DIR)

$(NOTEBOOKS_DIR): sanity_check_$(PYTHON_SCRIPTS_DIR) $(NOTEBOOKS_DIR)/*.ipynb sanity_check_$(NOTEBOOKS_DIR)

$(RENDERED_NOTEBOOKS_DIR): $(RENDERED_NOTEBOOKS_DIR)/*.ipynb sanity_check_$(RENDERED_NOTEBOOKS_DIR)

$(NOTEBOOKS_DIR)/%.ipynb:  $(PYTHON_SCRIPTS_DIR)/%.py
	jupytext --set-formats $(NOTEBOOKS_DIR)//ipynb,$(PYTHON_SCRIPTS_DIR)//py:percent $<
	jupytext --sync $<

$(RENDERED_NOTEBOOKS_DIR)/%.ipynb: $(NOTEBOOKS_DIR)/%.ipynb
	cp $< $@
	jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=None \
                    --ExecutePreprocessor.kernel_name=$(JUPYTER_KERNEL) --inplace $@

sanity_check_$(PYTHON_SCRIPTS_DIR):
	python scripts/check-python-scripts.py $(PYTHON_SCRIPTS_DIR)

sanity_check_$(NOTEBOOKS_DIR):
	python scripts/sanity-check.py $(PYTHON_SCRIPTS_DIR) $(NOTEBOOKS_DIR)

sanity_check_$(RENDERED_NOTEBOOKS_DIR):
	python scripts/sanity-check.py $(NOTEBOOKS_DIR) $(RENDERED_NOTEBOOKS_DIR)

