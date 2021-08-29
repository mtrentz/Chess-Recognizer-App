import cv2 as cv
from PIL import Image
import os
import numpy as np
import torch
from torchvision import transforms
from itertools import combinations
from collections import Counter
import io

class ChessRecognizer:
    def __init__(self, PIL_img):
        self.img = cv.cvtColor(np.array(PIL_img), cv.COLOR_RGB2BGR)
        self.img_gray = cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)
        self.Y_max, self.X_max, _ = self.img.shape
        self.HERE = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
        self.model = torch.load(os.path.join(self.HERE, 'torch_model.pth'), map_location=torch.device('cpu'))
        self.board_horizontal_lines, self.board_vertical_lines = self.board_lines(self.img_gray)
        self.board_squares = self.board_squares(self.board_horizontal_lines, self.board_vertical_lines)
        self.predicted_board = self.predict_board(self.img, self.model, self.board_squares)
    

    def _get_lines(self, img_gray):
        """Busca as linhas da imagem"""

        edges = cv.Canny(img_gray, 50, 150, apertureSize = 5)
        min_length = round(min(img_gray.shape[0], img_gray.shape[1])*0.08)
        max_gap = round(min(img_gray.shape[0], img_gray.shape[1])*0.01)
        lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=min_length, maxLineGap=max_gap)

        return lines

    def _get_oriented_lines(self, lines):
        """Separa as linhas horizontais/verticais"""

        # Quantos graus a linha pode desviar de completamente reta para ser considerada horizontal/vertical
        acceptable_angle_dev = 5

        # Vou separar as linhas em horizontais e verticais
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1,y1,x2,y2 = line[0]

            # Do jeito que ta colocado aqui, se for perfeitamente horizontal o angulo vai ser zero e vertical o angulo será -90
            horizontal_angle = 0
            vertical_angle = -90

            # Calcula angulo da linha
            angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))

            if (horizontal_angle - acceptable_angle_dev) <= angle <= (horizontal_angle + acceptable_angle_dev):
                horizontal_lines.append([x1,y1,x2,y2])
            elif (vertical_angle - acceptable_angle_dev) <= angle <= (vertical_angle + acceptable_angle_dev):
                vertical_lines.append([x1,y1,x2,y2])

        return horizontal_lines, vertical_lines
    
    def _extend_lines(self, horizontal_lines, vertical_lines):
        """Extende as linhas na imagem toda, cria também linhas horizontais/verticais nas bordas da img"""
        
        extended_horizontal_lines = [[0, l[1], self.X_max, l[3]] for l in horizontal_lines]
        extended_vertical_lines = [[l[0], 0, l[2], self.Y_max] for l in vertical_lines]

        ### remove duplicadas
        extended_horizontal_lines = np.unique(np.array(extended_horizontal_lines), axis=0).tolist()
        extended_vertical_lines = np.unique(np.array(extended_vertical_lines), axis=0).tolist()

        return extended_horizontal_lines, extended_vertical_lines

    def _get_line_gaps(self, extended_horizontal_lines, extended_vertical_lines):
        def h_line_gap(l1, l2):
            # Aqui se eu tenho duas linhas horizontais, eu vou ver o gap VERTICAL delas,
            # ou seja, o gap em Y
            delta_y1 = abs(l1[1] - l2[1])
            # Faz o mesmo pro y2
            delta_y2 = abs(l1[3] - l2[3])
            # Retorna a média dos dois
            return (delta_y1 + delta_y2)/2


        def v_line_gap(l1, l2):
            # Aqui se eu tenho duas linhas verticais, eu vou ver o gap horizontal delas,
            # ou seja, o gap em X
            delta_x1 = abs(l1[0] - l2[0])
            delta_x2 = abs(l1[2] - l2[2])
            return (delta_x1 + delta_x2)/2

        # Começo pegando os indexes (quantidade) de linhas horizontais e verticais
        h_line_indexes = [i for i in range(len(extended_horizontal_lines))]
        v_line_indexes = [i for i in range(len(extended_vertical_lines))]
        # Agora eu pego a combinação entre os indexes, que vou medir os gaps
        h_line_combinations = list(combinations(h_line_indexes, 2))
        v_line_combinations = list(combinations(v_line_indexes, 2))
        # Em cada combinação eu calculo o gap entre as linhas e salvo como (l1_index, l2_index, l1l2_gap)
        h_gaps = []
        v_gaps = []

        for h_comb in h_line_combinations:
            # h_comb[0] é o index de uma linha, h_comb[1] é o index de outra
            l1_index = h_comb[0]
            l2_index = h_comb[1]
            # Pego l1 e l2 em si
            l1 = extended_horizontal_lines[l1_index]
            l2 = extended_horizontal_lines[l2_index]
            # Calculo o gap entre elas
            hgap = h_line_gap(l1, l2)
            # Guardo os index e o gap entre elas
            h_gaps.append((l1_index, l2_index, hgap))

        for v_comb in v_line_combinations:
            l1_index = v_comb[0]
            l2_index = v_comb[1]
            l1 = extended_vertical_lines[l1_index]
            l2 = extended_vertical_lines[l2_index]
            vgap = v_line_gap(l1, l2)
            v_gaps.append((l1_index, l2_index, vgap))
        
        return h_gaps, v_gaps
    
    def _get_board_lines(self, extended_horizontal_lines, extended_vertical_lines, h_gaps, v_gaps):
        # Se a imagem fosse o board com nada de espaço nas bordas
        # cada célular ocuparia 1/8 da largura/altura da imagem
        # Assim, assumindo que as bordas não sejam muito grande,
        # vou assumir 1/7 da largura/altura como o maior gap possível
        # Lembrando que o gap entre duas linhas horizontais é o Y
        max_allowed_hline_gap = round(self.Y_max/7)
        max_allowed_vline_gap = round(self.X_max/7)
        # O menor gap vou assumir como 1/12 das dimensoes
        # caso contrario é pq o board foi muito mal focado na foto
        min_allowed_hline_gap = round(self.Y_max/12)
        min_allowed_vline_gap = round(self.X_max/12)

        # Pros gaps horizontal e vertical eu vou descobrir as frequencias dos gaps
        # aqui o g[2] é o valor do gap, visto que h_gaps[0] é [l1_ind, l2_ind, gap]
        h_gaps_freq = Counter(g[2] for g in h_gaps).most_common()
        v_gaps_freq = Counter(g[2] for g in v_gaps).most_common()

        # Agora filtro eles pra estarem dentro dos limites
        h_gaps_freq = [g for g in h_gaps_freq if min_allowed_hline_gap < g[0] < max_allowed_hline_gap]
        v_gaps_freq = [g for g in v_gaps_freq if min_allowed_vline_gap < g[0] < max_allowed_vline_gap]

        # O gap entre minhas celulas deve ser igual ao valor que apareceu com mais frequencia nas listas acima
        # ou pelo menos algum valor muito perto disso.
        wanted_v_gap = v_gaps_freq[0][0]
        wanted_h_gap = h_gaps_freq[0][0]
        # Vou agora iterar por todos gaps, se eles estiverem até N pixels de distância do que eu quero, vou separar aquela linha
        board_horizontal_lines = []
        board_vertical_lines = []
        allowed_threshold = 5

        # Agora eu itero pelos gaps horizontais, vejo se o gap está no limite aceitavel de 5 pixels
        for gap in h_gaps:
            if (wanted_h_gap - allowed_threshold) < gap[2] < (wanted_h_gap + allowed_threshold):
                # Caso esteja, eu dou append na l1 e l2 do gap, caso elas ainda não estejam no board_lines
                l1_index = gap[0]
                l2_index = gap[1]
                l1 = extended_horizontal_lines[l1_index]
                l2 = extended_horizontal_lines[l2_index]

                if l1 not in board_horizontal_lines:
                    board_horizontal_lines.append(l1)
                if l2 not in board_horizontal_lines:
                    board_horizontal_lines.append(l2)

        # Faço a mesma coisa pros verticais
        for gap in v_gaps:
            if (wanted_v_gap - allowed_threshold) < gap[2] < (wanted_v_gap + allowed_threshold):
                # Caso esteja, eu dou append na l1 e l2 do gap, caso elas ainda não estejam no board_lines
                l1_index = gap[0]
                l2_index = gap[1]
                l1 = extended_vertical_lines[l1_index]
                l2 = extended_vertical_lines[l2_index]

                if l1 not in board_vertical_lines:
                    board_vertical_lines.append(l1)
                if l2 not in board_vertical_lines:
                    board_vertical_lines.append(l2)

        # Tem a chance de eu ter pego mais board lines do que necessário
        # Assim, todas menos uma das linhas dentro de um threhshold devem ser removidas
        h_idx_to_remove = set()
        v_idx_to_remove = set()

        # Dps que iterei por todas as outras linhas em relação a uma e gravei as que tão perto dela
        # eu adiciono essa linha numa lista especial para não ser removida do board
        h_dont_remove = []
        v_dont_remove = []

        # Itera duas vezes pelas horizontais
        for ind1, h1 in enumerate(board_horizontal_lines):
            for ind2, h2 in enumerate(board_horizontal_lines):
                # Confere se as duas linhas tem um Y muito proximo um dos outros
                # Tem que ser maior que 1, pq se não qnd a linha compara com ela mesmo, vai remover
                if allowed_threshold >= abs(h1[1] - h2[1]) >= 1:
                    h_idx_to_remove.add(ind2)
                # Faz o mesmo pros Y2
                if allowed_threshold >= abs(h1[3] - h2[3]) >= 1:
                    h_idx_to_remove.add(ind2)

            if ind1 not in h_idx_to_remove:
                h_dont_remove.append(ind1)

        # Itera duas vezes pelas verticais
        for ind1, v1 in enumerate(board_vertical_lines):
            for ind2, v2 in enumerate(board_vertical_lines):
                # Confere se as duas linhas tem um X muito proximo um dos outros
                # Tem que ser maior que 1, pq se não qnd a linha compara com ela mesmo, vai remover
                if allowed_threshold>= abs(v1[0] - v2[0]) >= 1:
                    v_idx_to_remove.add(ind2)
                # Faz o mesmo pros X2
                if allowed_threshold>= abs(v1[2] - v2[2]) >= 1:
                    v_idx_to_remove.add(ind2)
            
            if ind1 not in v_idx_to_remove:
                v_dont_remove.append(ind1)

        h_idx_to_remove = list(h_idx_to_remove)
        v_idx_to_remove = list(v_idx_to_remove)

        # Realmente remove as linhas com aquele index do board_lines
        board_horizontal_lines = [v for i, v in enumerate(board_horizontal_lines) if i not in h_idx_to_remove or i in h_dont_remove]
        board_vertical_lines = [v for i, v in enumerate(board_vertical_lines) if i not in v_idx_to_remove or i in v_dont_remove]

        return board_horizontal_lines, board_vertical_lines
    
    def _get_intersections(self, board_horizontal_lines, board_vertical_lines):
        def make_line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def get_line_intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]
            if D != 0:
                x = Dx / D
                y = Dy / D
                return x,y
            else:
                return False

        intersections = []

        for hline in board_horizontal_lines:
            for vline in board_vertical_lines:
                intersec = get_line_intersection(make_line(hline[:2], hline[2:]),
                                            make_line(vline[:2], vline[2:]))
                intersections.append(intersec)

        intersections.sort()
        
        return intersections
    
    def _get_board_squares(self, intersections):
        board_points = np.array([
            [intersections[i] for i in range(0,9)],
            [intersections[i] for i in range(9,18)],
            [intersections[i] for i in range(18,27)],
            [intersections[i] for i in range(27,36)],
            [intersections[i] for i in range(36,45)],
            [intersections[i] for i in range(45,54)],
            [intersections[i] for i in range(54,63)],
            [intersections[i] for i in range(63,72)],
            [intersections[i] for i in range(72,81)],
        ])

        # Os quadrados do board são definido pelo (x1, y1) e (x2, y2) do topo esquerda e baixo direita
        board_squares = np.array([
            [(board_points[0][i], board_points[1][i+1])  for i in range(0,8)],
            [(board_points[1][i], board_points[2][i+1])  for i in range(0,8)],
            [(board_points[2][i], board_points[3][i+1])  for i in range(0,8)],
            [(board_points[3][i], board_points[4][i+1])  for i in range(0,8)],
            [(board_points[4][i], board_points[5][i+1])  for i in range(0,8)],
            [(board_points[5][i], board_points[6][i+1])  for i in range(0,8)],
            [(board_points[6][i], board_points[7][i+1])  for i in range(0,8)],
            [(board_points[7][i], board_points[8][i+1])  for i in range(0,8)],
        ])
        board_squares = board_squares.astype(int)

        return board_squares
    
    def board_lines(self, img_gray):

        lines = self._get_lines(img_gray)
        horizontal_lines, vertical_lines = self._get_oriented_lines(lines)
        extended_horizontal_lines, extended_vertical_lines = self._extend_lines(horizontal_lines, vertical_lines)
        h_gaps, v_gaps = self._get_line_gaps(extended_horizontal_lines, extended_vertical_lines)
        board_horizontal_lines, board_vertical_lines = self._get_board_lines(extended_horizontal_lines, extended_vertical_lines, h_gaps, v_gaps)

        return board_horizontal_lines, board_vertical_lines
    
    def board_squares(self, board_horizontal_lines, board_vertical_lines):

        intersections = self._get_intersections(board_horizontal_lines, board_vertical_lines)
        board_squares = self._get_board_squares(intersections)

        return board_squares
    
    def show_board_lines(self):
        img = self.img
        board_horizontal_lines = self.board_horizontal_lines
        board_vertical_lines = self.board_vertical_lines
        # Linhas horizontais em azul
        for line in board_horizontal_lines:
            x1,y1,x2,y2 = line
            cv.line(img,(x1,y1),(x2,y2),(255, 0, 0) ,2)

        # Linhas verticais em vermelho
        for line in board_vertical_lines:
            x1,y1,x2,y2 = line
            cv.line(img,(x1,y1),(x2,y2),(0, 0, 255) ,2)

        cv.imshow('board_lines', img)
        cv.waitKey(0)

        return None

    def predict_board(self, img, model, board_squares):
        # Classes do modelo
        class_map = {
            '0': 'bB',
            '1': 'bK',
            '2': 'bN',
            '3': 'bP',
            '4': 'bQ',
            '5': 'bR',
            '6': 'em',
            '7': 'wB',
            '8': 'wK',
            '9': 'wN',
            '10': 'wP',
            '11': 'wQ',
            '12': 'wR',
        }

        predicted_board = np.zeros((8,8)).astype(str)

        #cv.rectangle(img=img, pt1=tuple(board_squares[0][0][0]), pt2=tuple(board_squares[0][0][1]), color=(0,0,0), thickness=5)
        for col_index, col in enumerate(board_squares):
            for row_index, row in enumerate(col):

                data_transforms = transforms.Compose([
                        transforms.Resize((116, 116)),
                        transforms.ToTensor(),
                        ])

                cv_crop = img[row[0][1]:row[1][1], row[0][0]:row[1][0]]
                crop = Image.fromarray(cv_crop)
                crop = data_transforms(crop).unsqueeze(0)

                pred = model(crop)
                pred = pred.argmax(dim=1).numpy()

                #named_board_pieces[row_index, col_index] = class_map[str(int(pred[0]))]
                predicted_board[row_index, col_index] = class_map[str(int(pred[0]))]
        
        return predicted_board

def translate_pred_to_pt(predicted_board):
    pt_class_map = {
        'bB': 'Bispo - Preto',
        'bK': 'Rei - Preto',
        'bN': 'Cavalo - Preto',
        'bP': 'Peao - Preto',
        'bQ': 'Rainha - Preto',
        'bR': 'Torre - Preto',
        'em': 'Vazio',
        'wB': 'Bispo - Branco',
        'wK': 'Rei - Branco',
        'wN': 'Cavalo - Branco',
        'wP': 'Peao - Branco',
        'wQ': 'Rainha- Branco',
        'wR': 'Torre- Branco',
    }

    translated_board = np.vectorize(pt_class_map.get)(predicted_board)
    
    return translated_board

def translate_pred_to_unicode(predicted_board):

    code_map = {
        'bB': '\u265d',
        'bK': '\u265a',
        'bN': '\u265e',
        'bP': '\u265f',
        'bQ': '\u265b',
        'bR': '\u265c',
        'em': '',
        'wB': '\u2657',
        'wK': '\u2654',
        'wN': '\u2658',
        'wP': '\u2659',
        'wQ': '\u2655',
        'wR': '\u2656',
    }
    translated_board = np.vectorize(code_map.get)(predicted_board)

    return translated_board


def board_to_fen(board):
    # Coloca tudo em lowercase
    board = np.char.lower(board)

    # Use StringIO to build string more efficiently than concatenating
    with io.StringIO() as s:
        for row in board:
            empty = 0
            for cell in row:
                c = cell[0]
                if c in ('w', 'b'):
                    if empty > 0:
                        s.write(str(empty))
                        empty = 0
                    s.write(cell[1].upper() if c == 'w' else cell[1].lower())
                else:
                    empty += 1
            if empty > 0:
                s.write(str(empty))
            s.write('/')
        # Move one position back to overwrite last '/'
        s.seek(s.tell() - 1)
        # If you do not have the additional information choose what to put
        s.write(' w KQkq - 0 1')
        return s.getvalue()
        