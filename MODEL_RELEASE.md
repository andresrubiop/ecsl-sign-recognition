# Source release and model distribution

This is a **source-code release**. Aggregate results are in `results/`; no inference-ready weights are included or claimed to be publicly available here.

The original training pipeline uses two sources of materials with distinct terms:

1. Ecuadorian Sign Language source videos. Obtain them from their authorized publisher, verify their license and consent scope, and follow the dataset's procedures. The source project cites the [AI-Ready Ecuadorian Sign Language Corpus](https://doi.org/10.5281/zenodo.21001360).
2. An optional ASL Citizen pretrained model. Code adapted from Microsoft's MIT-licensed implementation retains its notice. The pretrained checkpoint and research materials have separate [Microsoft research terms](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/).

The public tree excludes `raw/`, `data/`, `artifacts/`, `weights/`, participant-level predictions, videos and checkpoints. Historical aggregate metrics are retained without participant identifiers or sample paths. An MIT source license does not authorize redistribution of separately licensed materials or derived model artefacts.

Before distributing target-model weights, establish the applicable distribution rights, choose a versioned release, validate hashes and loading behavior, and document provenance and dataset consent. Do not upload raw videos, processed landmarks or participant outcome logs as part of a model bundle.

ROS nodes, robot interfaces, STL designs and participant outcome data are outside this source release. Scripts for building a Jetson bundle do not establish that a public bundle exists or that target hardware was validated during release preparation.
