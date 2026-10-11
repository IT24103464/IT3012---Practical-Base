"""Optional additional deterministic proof that the A* integration avoids unsafe tiles."""
from agent import SearchAgent


unsafe = ['TargetVisible', 'HasDust', 'BloodseekerMissing']
safe = ['TargetVisible', 'HasDust']

agent = SearchAgent()
facts = {(1, 0): unsafe, (0, 1): safe}
path = agent.astar_search((0, 0), (2, 0), [], (3, 3), tile_facts=facts)
assert path == ['Up', 'Right', 'Right', 'Down'], f'Expected safe detour, got {path}'
print('A* avoids unsafe tile (1, 0). Safe detour:', ' → '.join(path))

blocked_goal = agent.astar_search((0, 0), (1, 0), [], (3, 3), tile_facts=facts)
assert blocked_goal is None, 'A* must not enter an unsafe goal.'
print('A* refuses a dangerous goal.')
