# Online colocated recipes

This directory is reserved so the catalog keeps the same mode/topology axes for
online and offline recipes. SpecForge does not currently implement online
colocated target capture, so no YAML belongs here. All supported online recipes
use separate producer and consumer roles under `../disaggregated/`, regardless
of whether one supervisor starts both roles.
