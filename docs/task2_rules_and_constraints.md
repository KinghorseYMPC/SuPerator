# Task 2 Rules And Constraints

Task 2 is the multi-physics Burgers prediction task. It is not implemented in the current project stage; SuPerator remains focused on Task 1 until a later phase explicitly opens Task 2 work.

## Rules

- Task 2 training data provides `Nu`.
- Task 2 test data does not provide `Nu`.
- Inference must predict only from the initial condition.
- Training time is not part of the Task 2 score, but total training duration must stay within 12 hours.
- Inference must finish in less than 2 minutes.
- Do not use Task 1 checkpoints.
- Do not use Task 1 data.
- Do not use pretrained models.
- Train from scratch.
- Do not use numerical solvers to generate extra training data.
- Use only official Task 2 data.

## Project Constraints Before Task 2 Work

Before entering Task 2 implementation, add explicit checks for:

- Task 2 data isolation from Task 1 data.
- Task 2 checkpoint isolation from Task 1 checkpoints.
- Task 2 config isolation from Task 1 configs.
- Training-from-scratch enforcement.
- Test-time input checks that prevent `Nu` from being used during inference.

No Task 2 model, training loop, or submission pipeline should be added until these isolation checks are designed and tested.
