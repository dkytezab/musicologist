# Future directions
As I see it, there are a few possible ways to go about achieving our goal: namely, observing how musical concepts emerge/change during the diffusion process. I'm listing two possibilities below.

# Higher-level concept extraction
Bear with me as GitHub's LaTeX support is limited. We create some diverse database $C$ of high-level music captions, e.g. "bright string orchestra" or "funky techno beat" which provides 
a reasonable representation of various genres/styles of music. For each of $t = 1, \dots, T$, we perform text-guided diffusion conditioned on our captions using $s_t$ backwards steps.
We then assemble $T$ batches of generated audio, i.e. $X_1, \dots, X_T$. We then pass each $X_i$ into our joint embedding model $J$ and create a data-label tuple $(J(X_i), C)$. We then train
either a simple MLP or a linear probe to predict $c \in C$ from $J(x) \in J(X_i)$.

The varying performance of these MLPs/probes on all sets $X_1, \dots, X_T$ then sheds some light on how high-level conceptual differences, e.g. genre and mood, emerge
through the diffusion process in a systematic fashion. If a given probe trained on $J(X_i)$ performs poorly, we expect that said conceptual differences aren't obviously present at
the $s_i$-th diffusion step.

# Low-level concept extraction
You'll note that the past methodology fails generally to uncover new high-level concepts not represented in the database $C$ and cannot
recover low-level musical components. To accomplish both of these aims, we rely on Concept Activation Vectors. Roughly speaking, CAVs can be used to interpret classifiers. They work by first gathering positive and negative samples, where the positive samples $A$ are clear representatives of some concept, say stripes for image classification, and negative samples $B$ are typically random noise. $A$ and $B$ are then passed into the classifier. We then extract their embeddings at the $l$-th layer $h_l$ and train a linear classifier (e.g. SVM) to separate $h_l(A)$ and $h_l(B)$. Our CAV is then the vector used in SVM, which we call $v_l$. Given some sample $x$, we can use $v_l$ to examine the sensitivity of the network's final prediction $f(x)$ to that original concept (more details in the original CAV paper).

For us, we first must assemble some set of fine-grain musical concepts and corresponding exemplar datasets. For instance, one concept might be "rhythmic drive" and consist of various beats at varying speeds. After passing our positive and negative datasets into $J(\cdot), and given $J(X_1), \dots, J(X_T)$ and our MLP classifiers $f_1, \dots, f_T$ paired as $(f_i, J(X_i))$, we can examine the sensitivity of $f_i$ to a given concept using the above approach. What's nice about this is that we can then cross compare $f_1, \dots, f_T$'s sensitivities to said concept (lets say rhythmic drive for now) for the entire dataset $J(X_i)$.