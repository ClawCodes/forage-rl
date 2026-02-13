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
        # State 1 stay
        assert maze.transitions[0].state == 0
        assert maze.transitions[0].action == 0
        assert maze.transitions[0].next_state == 0
        assert maze.transitions[0].prob == 1.0
        # State 1 leave
        assert maze.transitions[1].state == 0
        assert maze.transitions[1].action == 1
        assert maze.transitions[1].next_state == 1
        assert maze.transitions[1].prob == 1.0
        # State 2 stay
        assert maze.transitions[2].state == 1
        assert maze.transitions[2].action == 0
        assert maze.transitions[2].next_state == 1
        assert maze.transitions[2].prob == 1.0
        # State 2 leave
        assert maze.transitions[3].state == 1
        assert maze.transitions[3].action == 1
        assert maze.transitions[3].next_state == 0
        assert maze.transitions[3].prob == 1.0
