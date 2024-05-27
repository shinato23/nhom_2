import numpy as np
import matplotlib.pyplot as plt
import random

# Kích thước của Gridworld
grid_size = (5, 5)

# Các hành động có thể thực hiện
actions = ["up", "down", "left", "right"]

# Vị trí khởi đầu, mục tiêu, và cạm bẫy
start = (0, 0)
goal = (4, 4)
traps = [(0, 2), (0, 3), (1, 4), (2, 2), (3, 2),(3,4)]

# Phần thưởng cho mỗi trạng thái
def get_reward(state):
    if state == goal:
        return 1  # Phần thưởng khi đạt mục tiêu
    elif state in traps:
        return -1  # Phạt khi vào cạm bẫy
    else:
        return 0  # Phần thưởng trung lập

# Hàm để thực hiện hành động và trả về trạng thái mới
def take_action(state, action):
    x, y = state
    if action == "up":
        x = max(0, x - 1)
    elif action == "down":
        x = min(grid_size[0] - 1, x + 1)
    elif action == "left":
        y = max(0, y - 1)
    elif action == "right":
        y = min(grid_size[1] - 1, y + 1)
    return (x, y)

# Q-Table lưu trữ giá trị Q cho mỗi trạng thái và hành động
q_table = np.zeros((grid_size[0], grid_size[1], len(actions)))

# Tham số Q-Learning
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1  # Xác suất chọn hành động ngẫu nhiên (epsilon-greedy)

# Số tập (episodes) để huấn luyện
num_episodes = 1000

# Huấn luyện bằng Q-Learning
for episode in range(num_episodes):
    state = start  # Bắt đầu từ điểm khởi đầu
    while state != goal:  # Tiếp tục cho đến khi đạt mục tiêu
        # Epsilon-greedy: Chọn hành động ngẫu nhiên hoặc hành động tốt nhất
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)  # Chọn hành động ngẫu nhiên
        else:
            action = actions[np.argmax(q_table[state[0], state[1]])]  # Chọn hành động tốt nhất

        new_state = take_action(state, action)  # Thực hiện hành động
        reward = get_reward(new_state)  # Nhận phần thưởng
        old_q = q_table[state[0], state[1], actions.index(action)]  # Giá trị Q cũ
        best_future_q = np.max(q_table[new_state[0], new_state[1]])  # Giá trị Q tốt nhất trong tương lai

        # Cập nhật Q-Value theo công thức Q-Learning
        new_q = old_q + learning_rate * (reward + discount_factor * best_future_q - old_q)
        q_table[state[0], state[1], actions.index(action)] = new_q  # Cập nhật Q-Value

        state = new_state  # Cập nhật trạng thái hiện tại

# Hàm để hiển thị Gridworld với tác nhân, cạm bẫy, và mục tiêu
def display_grid(state):
    grid_display = np.zeros((grid_size[0], grid_size[1]))  # Lưới hiển thị

    # Đánh dấu vị trí của cạm bẫy
    for trap in traps:
        grid_display[trap[0], trap[1]] = -1  # Cạm bẫy

    # Đánh dấu vị trí của mục tiêu
    grid_display[goal[0], goal[1]] = 1  # Mục tiêu

    # Đánh dấu vị trí hiện tại của tác nhân
    grid_display[state[0], state[1]] = 0.5  # Tác nhân hiện tại

    plt.imshow(grid_display, cmap='gray', interpolation='nearest')  # Hiển thị lưới
    plt.title(f"Gridworld - Current State: {state}")
    plt.show()

# Di chuyển tác nhân theo Q-Table và hiển thị từng bước
state = start  # Bắt đầu từ điểm khởi đầu
steps = 0  # Số bước để đạt mục tiêu

while state != goal:  # Di chuyển cho đến khi đạt mục tiêu
    action = actions[np.argmax(q_table[state[0], state[1]])]  # Chọn hành động tốt nhất
    state = take_action(state, action)  # Thực hiện hành động

    # In ra vị trí hiện tại
    print(f"Vị trí hiện tại: {state}")

    # Hiển thị lưới với trạng thái hiện tại
    display_grid(state)

    steps += 1  # Đếm số bước

print(f"Đến mục tiêu sau {steps} bước")
