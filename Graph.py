import vk_api
import networkx as nx
import pandas as pd
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import time

# Sidebar configuration
st.sidebar.subheader("Настройки графа")
width = st.sidebar.slider("Ширина", 500, 1200, 950)
height = st.sidebar.slider("Высота", 500, 1200, 700)
directed = st.sidebar.checkbox("Направленный граф", True)
physics = st.sidebar.checkbox("Физика", True)
hierarchical = st.sidebar.checkbox("Иерархический", False)
max_depth = st.sidebar.slider("Максимальная глубина", 1, 2, 1)  # Default depth is 1
max_friends = st.sidebar.slider("Максимальное количество друзей на пользователя", 50, 500, 100)  # Limit friends per user

# VK API Access Token
VK_ACCESS_TOKEN = '2ff60f692ff60f692ff60f695e2ce4eed822ff62ff60f694c094cb35bc606f13492ff84'

# List of VK IDs and full names (Your group members)
group_members = [
    {'VK ID': 290530655, 'ФИО': 'Алимов Исмаил Рифатович'},
    {'VK ID': None, 'ФИО': 'Баклашкин Алексей Андреевич'},
    {'VK ID': 1931147, 'ФИО': 'Брежнев Вячеслав Александрович'},
    {'VK ID': 207227130, 'ФИО': 'Волков Матвей Андреевич'},
    {'VK ID': None, 'ФИО': 'Гаев Роман Алексеевич'},
    {'VK ID': None, 'ФИО': 'Кирьянов Павел Александрович'},
    {'VK ID': 138042735, 'ФИО': 'Кравцов Кирилл Егорович'},
    {'VK ID': 172244589, 'ФИО': 'Лавренченко Мария Кирилловна'},
    {'VK ID': 168420440, 'ФИО': 'Лагуткина Мария Сергеевна'},
    {'VK ID': 711398942, 'ФИО': 'Лыгун Кирилл Андреевич'},
    {'VK ID': None, 'ФИО': 'Орачев Алексей Валерьевич'},
    {'VK ID': None, 'ФИО': 'Панарин Родион Владимирович'},
    {'VK ID': 65657314, 'ФИО': 'Пешков Максим Юрьевич (староста)'},
    {'VK ID': None, 'ФИО': 'Прозоров Евгений Иванович'},
    {'VK ID': 50933461, 'ФИО': 'Свинаренко Владислав Александрович'},
    {'VK ID': None, 'ФИО': 'Союзов Владимир Александрович'},
    {'VK ID': 198216820, 'ФИО': 'Хренникова Ангелина Сергеевна'},
    {'VK ID': None, 'ФИО': 'Черкасов Егор Юрьевич'},
    {'VK ID': 268235974, 'ФИО': 'Яминов Руслан Вильевич'}
]

# Extract VK IDs, excluding None values
vk_ids = [member['VK ID'] for member in group_members if member['VK ID'] is not None]

# Create a mapping from VK ID to full name for group members
group_vkid_to_name = {member['VK ID']: member['ФИО'] for member in group_members if member['VK ID'] is not None}

st.write("## Граф социальной сети VK")

def get_vk_session():
    session = vk_api.VkApi(token=VK_ACCESS_TOKEN)
    vk = session.get_api()
    return vk

def build_vk_graph(vk_ids, max_depth=1, max_friends=100):
    vk = get_vk_session()
    G = nx.Graph()

    processed_ids = set()
    queue = [(vk_id, 0) for vk_id in vk_ids]  # Initialize queue with initial VK IDs and depth 0

    private_profiles = set()  # To keep track of private profiles
    start_time = time.time()

    while queue:
        vk_id, depth = queue.pop(0)
        if vk_id in processed_ids:
            continue
        processed_ids.add(vk_id)

        G.add_node(vk_id)

        if depth >= max_depth:
            continue

        try:
            # Get user's friends with a limit
            response = vk.friends.get(user_id=vk_id, count=max_friends)
            friends = response.get('items', [])
            time.sleep(0.34)  # Sleep to respect VK API rate limits (max 3 requests/sec)

            for friend_id in friends:
                if friend_id not in G:
                    G.add_node(friend_id)
                G.add_edge(vk_id, friend_id)

                if friend_id not in processed_ids:
                    queue.append((friend_id, depth + 1))  # Include friends of friends up to max_depth

        except vk_api.exceptions.ApiError as e:
            error_code = e.code
            if error_code == 30:
                private_profiles.add(vk_id)
            else:
                st.write(f"Ошибка при получении друзей для VK ID {vk_id}: {e}")
            continue

        # Optional: Early stopping if computation takes too long
        if time.time() - start_time > 300:  # 5 minutes timeout
            st.warning("Время вычислений превышает лимит в 5 минут. Прерывание сбора данных.")
            break

    if private_profiles:
        st.info(f"Обнаружено {len(private_profiles)} приватных профилей. Их друзья недоступны.")

    return G

def convert_graph_to_streamlit_format(G, vk_ids, group_vkid_to_name):
    vk = get_vk_session()
    nodes = []
    edges = []

    # Calculate centrality measures for the entire graph
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

    # Fetch user info for all unique user IDs in the graph (excluding group members)
    user_ids = [node for node in G.nodes if node not in group_vkid_to_name]
    user_info_dict = {}
    batch_size = 1000  # VK API allows up to 1000 user IDs per request

    for i in range(0, len(user_ids), batch_size):
        batch_ids = user_ids[i:i+batch_size]
        try:
            users_info = vk.users.get(user_ids=batch_ids)
            time.sleep(0.34)  # Sleep to respect VK API rate limits
            for user_info in users_info:
                uid = user_info['id']
                # For friends and friends of friends, we only use VK ID
                user_info_dict[uid] = f"VK ID {uid}"
        except vk_api.exceptions.ApiError as e:
            # Handle errors gracefully
            for uid in batch_ids:
                user_info_dict[uid] = f"VK ID {uid}"  # Fallback to VK ID
            continue

    # Create nodes and edges for visualization
    for node in G.nodes:
        if node in group_vkid_to_name:
            # For group members, use full names
            full_name = group_vkid_to_name[node]
            profile_url = f"https://vk.com/id{node}"
            # Include centrality info
            centrality_info = (
                f"Степень центральности: {degree_centrality[node]:.4f}\n"
                f"Центральность посредничества: {betweenness_centrality[node]:.4f}\n"
                f"Центральность близости: {closeness_centrality[node]:.4f}\n"
                f"Собственный вектор центральности: {eigenvector_centrality[node]:.4f}\n"
                f"Профиль: {profile_url}"
            )
        else:
            # For others, use VK ID as label
            full_name = user_info_dict.get(node, f"VK ID {node}")
            centrality_info = ""
            profile_url = f"https://vk.com/id{node}"

        nodes.append(Node(
            id=str(node),
            label=full_name,
            size=25,
            title=centrality_info
        ))

    for source, target in G.edges:
        edges.append(Edge(source=str(source), target=str(target)))

    # Prepare centrality measures dataframe for group members
    centrality_data = []
    for vk_id in vk_ids:
        if vk_id in G.nodes:
            full_name = group_vkid_to_name.get(vk_id, f"VK ID {vk_id}")
            centrality_data.append({
                'VK ID': vk_id,
                'ФИО': full_name,
                'Степень центральности': degree_centrality.get(vk_id, 0),
                'Центральность посредничества': betweenness_centrality.get(vk_id, 0),
                'Центральность близости': closeness_centrality.get(vk_id, 0),
                'Собственный вектор центральности': eigenvector_centrality.get(vk_id, 0),
            })

    centrality_df = pd.DataFrame(centrality_data)
    # Round the centrality measures for better readability
    centrality_df[['Степень центральности', 'Центральность посредничества',
                   'Центральность близости', 'Собственный вектор центральности']] = centrality_df[[
        'Степень центральности', 'Центральность посредничества',
        'Центральность близости', 'Собственный вектор центральности']].round(4)

    return nodes, edges, centrality_df

# Build the VK graph including friends of friends
graph = build_vk_graph(vk_ids, max_depth=max_depth, max_friends=max_friends)

# Check if initial VK IDs are connected
components = list(nx.connected_components(graph))
node_to_component = {}
for idx, component in enumerate(components):
    for node in component:
        node_to_component[node] = idx

initial_components = set(node_to_component.get(vk_id) for vk_id in vk_ids if vk_id in node_to_component)

if len(initial_components) > 1:
    st.warning("Внимание: Нет связи между заданными VK ID даже через друзей друзей.")
else:
    st.success("Все заданные VK ID связаны через друзей или друзей друзей.")

# Convert the graph to the format compatible with streamlit-agraph and get centrality dataframe
nodes, edges, centrality_df = convert_graph_to_streamlit_format(graph, vk_ids, group_vkid_to_name)

config = Config(
    width=width,
    height=height,
    directed=directed,
    physics=physics,
    hierarchical=hierarchical
)

# Create the agraph using streamlit-agraph
return_value = agraph(nodes=nodes, edges=edges, config=config)

# Display the centrality measures dataframe below the graph
st.write("### Центральность участников группы")
st.dataframe(centrality_df)
