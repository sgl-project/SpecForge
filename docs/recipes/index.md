---
title: Recipes
description: End-to-end reproductions of real SpecForge training runs, with the exact source revisions, configs and launch commands.
prev: false
next: false
---

# Recipes

Recipes are complete, reproducible training runs: the exact source revisions,
configs, node layout and launch commands used to train a draft model that ships
in [SpecBundle](../web/specbundle.md). Where the guides explain each option,
a recipe shows one proven combination end to end.

<RecipeIndex />

## Contribute a recipe

Trained a draft model that others should be able to reproduce? Add a Markdown
file to
[`docs/recipes/`](https://github.com/sgl-project/SpecForge/tree/main/docs/recipes)
with a `title`, `description`, `target`, `method` and `topology` in its
frontmatter and open a pull request. The card above and the sidebar entry are
generated from the file, so nothing else needs to change. See the
[docs guide](https://github.com/sgl-project/SpecForge/blob/main/docs/README.md#add-a-recipe)
for the template.
