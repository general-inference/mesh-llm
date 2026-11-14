# Maintainer decisions

## Retain the external AgentX client

The maintainer decision approved retaining the pinned upstream client and
locked environment, with setup using an installed Python 3.12 interpreter.

The bounded owner is `skippy/evals/agentx/`: its exact upstream AgentX wheel,
dependency manifest, lock and setup invocation are registered in
`ci/python-tooling-policy.json`. Setup does not download an interpreter.
Campaign workers consume the prepared environment offline and verify its bytes.
This decision permits the external client dependency; it does not permit new
repository Python implementations or a general interpreter invocation bypass.
Changing the admitted inputs still requires updating their exact typed policy
and passing the policy's rejection tests and normal repository consistency gate.
