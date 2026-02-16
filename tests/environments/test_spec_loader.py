from forage_rl.environments.spec_loader import load_builtin_maze_spec


class TestSpecLoader:
    def test_load_simple_maze(self):
        # Arrange & Act
        maze = load_builtin_maze_spec(name="simple")

        # Assert
        # Meta info
        assert maze.maze.name == "simple"
        assert maze.maze.horizon == 100
        assert maze.maze.initial_state == 0
        assert maze.maze.action_labels == ["stay", "leave"]

        # States
        assert len(maze.states) == 2
        assert maze.states[0].id == 0
        assert maze.states[0].label == "Upper Patch"
        assert maze.states[0].decay == 0.2
        assert maze.states[0].observation_group == 0

        assert maze.states[1].id == 1
        assert maze.states[1].label == "Lower Patch"
        assert maze.states[1].decay == 3.0
        assert maze.states[1].observation_group == 1

        # Transitions
        assert len(maze.transitions) == 4
        # State 0 stay
        assert maze.transitions[0].state == 0
        assert maze.transitions[0].action == 0
        assert maze.transitions[0].next_state == 0
        assert maze.transitions[0].prob == 1.0
        # State 0 leave
        assert maze.transitions[1].state == 0
        assert maze.transitions[1].action == 1
        assert maze.transitions[1].next_state == 1
        assert maze.transitions[1].prob == 1.0
        # State 1 stay
        assert maze.transitions[2].state == 1
        assert maze.transitions[2].action == 0
        assert maze.transitions[2].next_state == 1
        assert maze.transitions[2].prob == 1.0
        # State 1 leave
        assert maze.transitions[3].state == 1
        assert maze.transitions[3].action == 1
        assert maze.transitions[3].next_state == 0
        assert maze.transitions[3].prob == 1.0

    def test_load_full_maze(self):
        # Arrange & Act
        maze = load_builtin_maze_spec(name="full")

        # Assert
        # Meta info
        assert maze.maze.name == "full"
        assert maze.maze.horizon == 100
        assert maze.maze.initial_state == 0
        assert maze.maze.action_labels == ["stay", "leave"]

        # States
        assert len(maze.states) == 6
        assert maze.states[0].id == 0
        assert maze.states[0].label == "Upper Patch"
        assert maze.states[0].decay == 0.5
        assert maze.states[0].observation_group == 0

        assert maze.states[1].id == 1
        assert maze.states[1].label == "Upper Patch"
        assert maze.states[1].decay == 2.0
        assert maze.states[1].observation_group == 0

        assert maze.states[2].id == 2
        assert maze.states[2].label == "Upper Patch"
        assert maze.states[2].decay == 0.1
        assert maze.states[2].observation_group == 0

        assert maze.states[3].id == 3
        assert maze.states[3].label == "Lower Patch"
        assert maze.states[3].decay == 0.1
        assert maze.states[3].observation_group == 1

        assert maze.states[4].id == 4
        assert maze.states[4].label == "Lower Patch"
        assert maze.states[4].decay == 2.0
        assert maze.states[4].observation_group == 1

        assert maze.states[5].id == 5
        assert maze.states[5].label == "Lower Patch"
        assert maze.states[5].decay == 3.0
        assert maze.states[5].observation_group == 1

        # Transitions
        assert len(maze.transitions) == 24
        # State 0 stay
        assert maze.transitions[0].state == 0
        assert maze.transitions[0].action == 0
        assert maze.transitions[0].next_state == 0
        assert maze.transitions[0].prob == 1.0
        # State 0 leave
        assert maze.transitions[1].state == 0
        assert maze.transitions[1].action == 1
        assert maze.transitions[1].next_state == 3
        assert maze.transitions[1].prob == 0.15

        assert maze.transitions[2].state == 0
        assert maze.transitions[2].action == 1
        assert maze.transitions[2].next_state == 4
        assert maze.transitions[2].prob == 0.35

        assert maze.transitions[3].state == 0
        assert maze.transitions[3].action == 1
        assert maze.transitions[3].next_state == 5
        assert maze.transitions[3].prob == 0.50
        # State 3 stay
        assert maze.transitions[12].state == 3
        assert maze.transitions[12].action == 0
        assert maze.transitions[12].next_state == 3
        assert maze.transitions[12].prob == 1.0
        # State 3 leave
        assert maze.transitions[13].state == 3
        assert maze.transitions[13].action == 1
        assert maze.transitions[13].next_state == 0
        assert maze.transitions[13].prob == 0.15

        assert maze.transitions[14].state == 3
        assert maze.transitions[14].action == 1
        assert maze.transitions[14].next_state == 1
        assert maze.transitions[14].prob == 0.35

        assert maze.transitions[15].state == 3
        assert maze.transitions[15].action == 1
        assert maze.transitions[15].next_state == 2
        assert maze.transitions[15].prob == 0.50
