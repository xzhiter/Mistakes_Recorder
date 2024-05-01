import os
import pickle
import sys
import traceback
from io import BytesIO

import fitz
import numpy as np
import pygame
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QFileDialog, QApplication, QMainWindow, QMessageBox, QDialog
from aip import AipOcr
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import ocr
import erase
import settings
from win import Ui_MainWindow


class TextRect:
    def __init__(self, inf, rate):
        self.text = inf["words"]
        self.flag = -1
        location = inf["location"]
        self.cut_rect = pygame.Rect(location["left"], location["top"], location["width"], location["height"])
        self.rect = pygame.Rect(int(self.cut_rect.x * rate), int(self.cut_rect.y * rate), int(self.cut_rect.w * rate),
                                int(self.cut_rect.h * rate))


class QuestionRect:
    def __init__(self, rect, rate, surface):
        self.cut_rect = [rect]
        self.rect = []
        self.touched = -1
        self.selected = False
        self.rate = rate
        self.surface = surface

    def draw(self):
        if self.touched > -1:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        for _i in self.rect:
            pygame.draw.rect(self.surface, color, _i, 1)
            if self.selected:
                surface = pygame.Surface((_i.w - 2, _i.h - 2), pygame.SRCALPHA)
                surface.fill((0, 255, 0, 100))
                self.surface.blit(surface, (_i.x + 1, _i.y + 1))

    def done(self):
        self.rect = []
        for _i in self.cut_rect:
            self.rect.append(pygame.Rect(int(_i.x * self.rate), int(_i.y * self.rate), int(_i.w * self.rate),
                                         int(_i.h * self.rate)))

    def sense(self):
        self.touched = -1
        for _i in range(len(self.rect)):
            if self.rect[_i].collidepoint(pygame.mouse.get_pos()):
                self.touched = _i


def pdf(pdf_path, image_path):
    pdf_doc = fitz.open(pdf_path)
    for pg in range(pdf_doc.page_count):
        pdf_page = pdf_doc[pg]
        rotate = int(0)
        zoom_x = 5
        zoom_y = 5
        mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        pix = pdf_page.get_pixmap(matrix=mat, alpha=False)
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        if "__single__section__XZHRO" in pdf_path:
            last = "__single__section__XZHRO"
        else:
            last = ""
        pix._writeIMG(f"{image_path}/{pg}{last}.png", format_=1, jpg_quality=None)


class Page:
    def __init__(self, path, surface, pages):
        if os.path.splitext(path)[1] == ".pdf":
            pdf(path, "./__ache__")
            images = os.listdir("./__ache__")
            for p in images:
                Page("./__ache__" + "\\" + p, surface, pages)
            return
        f = open(path, "rb")
        file_bin = f.read()
        result = client.accurate(file_bin)["words_result"]
        image = pygame.image.load(BytesIO(file_bin))
        self.pil_image = Image.open(BytesIO(file_bin))
        f.close()
        del file_bin
        size = image.get_size()
        rate = screen_size[0] / size[0]
        if size[1] * rate > screen_size[1]:
            rate = screen_size[1] / size[1]
        self.rate = rate
        self.image = pygame.transform.smoothscale(image,
                                                  (int(image.get_width() * rate), int(image.get_height() * rate)))
        self.rects = []
        pos = []
        for _i in result:
            if len(_i["words"]) > -1:
                rect = TextRect(_i, rate)
                pos.append([rect.cut_rect.center[0], 0])
                self.rects.append(rect)
        x = np.array(pos)
        if "__single__section__XZHRO" not in path:
            scores = []
            means = []
            for _k in range(2, 5, 1):
                k_means = KMeans(n_clusters=_k).fit(x)
                scores.append(silhouette_score(x, k_means.labels_, metric="euclidean"))
                means.append(k_means)
            self.section_num = scores.index(max(scores)) + 2
            k_means = means[self.section_num - 2]
        else:
            self.section_num = 1
            k_means = KMeans(n_clusters=1).fit(x)
        centers = k_means.cluster_centers_.copy()
        labels = list(range(self.section_num))
        for _i in range(self.section_num - 1):
            for _j in range(self.section_num - 1):
                if centers[_j][0] > centers[_j + 1][0]:
                    labels[_j], labels[_j + 1] = labels[_j + 1], labels[_j]
                    centers[_j][0], centers[_j + 1][0] = centers[_j + 1][0], centers[_j][0]
        centers = k_means.cluster_centers_
        for _i in range(len(k_means.labels_)):
            flag = k_means.labels_[_i]
            if abs(self.rects[_i].cut_rect.center[0] - centers[flag][0]) < size[0] / self.section_num * 0.33 and \
                    self.rects[_i].cut_rect.w < size[0] / self.section_num:
                self.rects[_i].flag = labels.index(flag)
        for _i in range(len(self.rects) - 1):
            for _j in range(len(self.rects) - 1):
                if self.rects[_j].flag > self.rects[_j + 1].flag:
                    self.rects[_j], self.rects[_j + 1] = self.rects[_j + 1], self.rects[_j]
        section_rect = []
        last = -1
        for _i in self.rects:
            if _i.flag == -1:
                continue
            if _i.flag != last:
                last = _i.flag
                section_rect.append(_i.cut_rect.copy())
            if section_rect[last].x > _i.cut_rect.x:
                section_rect[last].w += section_rect[last].x - _i.cut_rect.x
                section_rect[last].x = _i.cut_rect.x
            if section_rect[last].w + section_rect[last].x < _i.cut_rect.w + _i.cut_rect.x:
                section_rect[last].w = _i.cut_rect.w + _i.cut_rect.x - section_rect[last].x
            section_rect[last].h = _i.cut_rect.y + _i.cut_rect.h - section_rect[last].y
        self.section_rects = section_rect
        num = ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.", "11.", "12.", "13.", "14.", "15.", "16.",
               "17.", "18.", "19.", "20.", "21.", "22.")
        question_rects = []
        _item = -1
        section = -1
        rect_item = -1
        for _i in self.rects:
            if _i.flag == -1:
                continue
            for _j in num:
                if _i.text.startswith(_j):
                    if len(_i.text) != len(_j):
                        _i.text += "     "
                    if len(_i.text) == len(_j):
                        cut = True
                    elif not _i.text[len(_j)] in "0123456789":
                        cut = True
                    elif _i.text[len(_j) + 4] == "年" or _i.text[len(_j) + 1] == "分" or _i.text[len(_j) + 2] == "分"\
                            or _i.text[len(_j) + 3] == "分":
                        cut = True
                    else:
                        cut = False
                    if cut:
                        _item += 1
                        question_rects.append(QuestionRect(_i.cut_rect.copy(), rate, surface))
                        section = _i.flag
                        rect_item = 0
                        continue
            if _item == -1:
                continue
            if _i.flag != section:
                rect_item += 1
                section = _i.flag
                question_rects[_item].cut_rect.append(_i.cut_rect.copy())
            if question_rects[_item].cut_rect[rect_item].x > _i.cut_rect.x:
                question_rects[_item].cut_rect[rect_item].w += question_rects[_item].cut_rect[
                                                                   rect_item].x - _i.cut_rect.x
                question_rects[_item].cut_rect[rect_item].x = _i.cut_rect.x
            if question_rects[_item].cut_rect[rect_item].w + question_rects[_item].cut_rect[rect_item].x < \
                    _i.cut_rect.w + _i.cut_rect.x:
                question_rects[_item].cut_rect[rect_item].w = _i.cut_rect.w + _i.cut_rect.x - \
                                                              question_rects[_item].cut_rect[rect_item].x
            question_rects[_item].cut_rect[rect_item].h = _i.cut_rect.y + _i.cut_rect.h - \
                                                          question_rects[_item].cut_rect[rect_item].y
        for _i in question_rects:
            _i.done()
        self.question_rects = question_rects
        pages.append(self)


def process(dir_path, sub):
    sub_index = index[sub]
    pages = []
    err = []
    item = 0
    paths = os.listdir(dir_path)
    file_item = 0
    pygame.init()
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("错题整理", "错题整理")
    for i in paths:
        file_item += 1
        Page(dir_path + "\\" + i, screen, pages)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                document = Document()
                docx_section = document.sections[0]
                docx_section.page_width = Cm(18.2)
                docx_section.page_height = Cm(25.7)
                docx_section.top_margin = Cm(0.3)
                docx_section.right_margin = Cm(0.9)
                docx_section.bottom_margin = Cm(1.3)
                docx_section.left_margin = Cm(1.4)
                sect_pr = docx_section._sectPr
                cols = sect_pr.xpath("./w:cols")[0]
                cols.set(qn("w:num"), "2")
                cols.set(qn("w:space"), "20")
                for k in pages:
                    for i in k.question_rects:
                        if i.selected:
                            img_size = [0, 0]
                            for j in i.cut_rect:
                                if j.w > img_size[0]:
                                    img_size[0] = j.w
                                img_size[1] += j.h
                            new = Image.new("RGB", (img_size[0], img_size[1]), (255, 255, 255))
                            y = 0
                            for j in i.cut_rect:
                                new.paste(k.pil_image.crop((j.x, j.y, j.x + j.w, j.y + j.h)), (0, y))
                                y += j.h
                            img_byte = BytesIO()
                            new.save(img_byte, format="PNG")
                            num = len(sub_index)
                            with open(f"./{sub}/{num}.png", "wb") as file:
                                info = {
                                    "path": f"./{sub}/{num}.png",
                                    "removed": f"./{sub}/{num}_removed.png",
                                    "num": 0,
                                    "processed": False
                                }
                                sub_index.append(info)
                                file.write(img_byte.getvalue())
                            if options["erase"]:
                                with open(f"./{sub}/{num}_removed.png", "wb") as file:
                                    file.write(erase.create_request(f"./{sub}/{num}.png"))
                            document.add_picture(img_byte, width=Cm(7.7))
                            # document.add_picture(img_byte, width=Cm(15.9))
                with open(f"./{sub}/index.pkl", "wb") as file:
                    pickle.dump(sub_index, file)
                pygame.quit()
                return document
            if event.type == pygame.MOUSEBUTTONDOWN:
                for r in pages[item].question_rects:
                    if r.touched > -1:
                        r.selected = not r.selected
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    item += 1
                    if item >= len(pages):
                        item = 0
                if event.key == pygame.K_LEFT:
                    item -= 1
                    if item < 0:
                        item = len(pages) - 1
                if event.key == pygame.K_DELETE:
                    for r in pages[item].question_rects:
                        if r.touched > -1:
                            del r.cut_rect[r.touched]
                            del r.rect[r.touched]
                if event.key == pygame.K_a:
                    for r in pages[item].question_rects:
                        if r.touched > -1:
                            r.cut_rect[r.touched].x -= 10
                            r.cut_rect[r.touched].w += 10
                            r.done()
                if event.key == pygame.K_w:
                    for r in pages[item].question_rects:
                        if r.touched > -1:
                            r.cut_rect[r.touched].y -= 10
                            r.cut_rect[r.touched].h += 10
                            r.done()
                if event.key == pygame.K_d:
                    for r in pages[item].question_rects:
                        if r.touched > -1:
                            r.cut_rect[r.touched].w += 10
                            r.done()
                if event.key == pygame.K_s:
                    for r in pages[item].question_rects:
                        if r.touched > -1:
                            r.cut_rect[r.touched].h += 10
                            r.done()
                if event.key == pygame.K_l:
                    for r in pages[item].question_rects:
                        if r.touched > -1:
                            r.cut_rect[r.touched].x += 10
                            r.cut_rect[r.touched].w -= 10
                            r.done()
                if event.key == pygame.K_k:
                    for r in pages[item].question_rects:
                        if r.touched > -1:
                            r.cut_rect[r.touched].y += 10
                            r.cut_rect[r.touched].h -= 10
                            r.done()
                if event.key == pygame.K_j:
                    for r in pages[item].question_rects:
                        if r.touched > -1:
                            r.cut_rect[r.touched].w -= 10
                            r.done()
                if event.key == pygame.K_i:
                    for r in pages[item].question_rects:
                        if r.touched > -1:
                            r.cut_rect[r.touched].h -= 10
                            r.done()
        screen.fill((255, 255, 255))
        screen.blit(pages[item].image, (0, 0))
        for r in pages[item].question_rects:
            r.sense()
            r.draw()
        pygame.display.update()


def err_handler(path, sub):
    try:
        return process(path, sub)
    except:
        return traceback.format_exc()


class Thread(QThread):
    signal_tuple = pyqtSignal(tuple)

    def __init__(self, func):
        super().__init__()
        self.func = func

    def run(self):
        result = self.func()
        self.signal_tuple.emit((result, 1))


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent=parent)
        self.setupUi(self)
        self.actionOpen.triggered.connect(lambda: self.setup_thread(None))
        self.pushButtonMath.clicked.connect(lambda: self.setup_thread("math"))
        self.pushButtonPhysics.clicked.connect(lambda: self.setup_thread("phy"))
        self.pushButtonChem.clicked.connect(lambda: self.setup_thread("chem"))
        self.pushButtonBio.clicked.connect(lambda: self.setup_thread("bio"))
        self.actionSettings.triggered.connect(self.settings)

    def setup_thread(self, sub):
        dir_path = QFileDialog.getExistingDirectory(win, "浏览", "C:/")
        self.thread_ = Thread(lambda: err_handler(dir_path, sub))
        self.thread_.signal_tuple.connect(self.thread_finished)
        self.thread_.start()

    @pyqtSlot(tuple)
    def thread_finished(self, item):
        if isinstance(item[0], str):
            QMessageBox.critical(win, "程序运行出错", f"请将以下信息拍照发给开发者：\n{item[0]}")
        else:
            item[0].save(QFileDialog.getSaveFileName(win, "保存", "C:/", "Word 文档 (*.docx)")[0])

    def settings(self):
        class Settings(QDialog, settings.Ui_Dialog):
            def __init__(self, parent=win):
                QDialog.__init__(self, parent)
                self.setupUi(self)
                self.lineEdit.setText(str(options["size"][0]))
                self.lineEdit_2.setText(str(options["size"][1]))
                self.checkBox.setChecked(options["online"])
                self.lineEdit_3.setText(options["APP_ID"])
                self.lineEdit_4.setText(options["API_KEY"])
                self.lineEdit_5.setText(options["SECRET_KEY"])
                self.checkBox_2.setChecked(options["erase"])
                self.lineEdit_6.setText(options["yd_id"])
                self.lineEdit_7.setText(options["yd_key"])
        settings_window = Settings()
        settings_window.move(win.pos())
        settings_window.exec()
        global options
        options = {
            "size": (int(settings_window.lineEdit.text()), int(settings_window.lineEdit_2.text())),
            "online": settings_window.checkBox.isChecked(),
            "APP_ID": settings_window.lineEdit_3.text(),
            "API_KEY": settings_window.lineEdit_4.text(),
            "SECRET_KEY": settings_window.lineEdit_5.text(),
            "erase": settings_window.checkBox_2.isChecked(),
            "yd_id": settings_window.lineEdit_6.text(),
            "yd_key": settings_window.lineEdit_7.text(),
        }
        with open("./options.pkl", "wb") as f_:
            pickle.dump(options, f_)


if __name__ == "__main__":
    if os.path.isfile("./options.pkl"):
        with open("./options.pkl", "rb") as f:
            options = pickle.load(f)
    else:
        options = {
            "size": (1500, 900),
            "online": False,
            "APP_ID": "",
            "API_KEY": "",
            "SECRET_KEY": "",
            "erase": False,
            "yd_id": "",
            "yd_secret": ""
        }
        with open("./options.pkl", "wb") as f:
            pickle.dump(options, f)
    if not os.path.isdir("./math"):
        os.mkdir("./math")
    if os.path.isfile("./math/index.pkl"):
        with open("./math/index.pkl", "rb") as f:
            math_index = pickle.load(f)
    else:
        math_index = []
        with open("./math/index.pkl", "wb") as f:
            pickle.dump(math_index, f)
    if not os.path.isdir("./phy"):
        os.mkdir("./phy")
    if os.path.isfile("./phy/index.pkl"):
        with open("./phy/index.pkl", "rb") as f:
            phy_index = pickle.load(f)
    else:
        phy_index = []
        with open("./phy/index.pkl", "wb") as f:
            pickle.dump(phy_index, f)
    if not os.path.isdir("./chem"):
        os.mkdir("./chem")
    if os.path.isfile("./chem/index.pkl"):
        with open("./chem/index.pkl", "rb") as f:
            chem_index = pickle.load(f)
    else:
        chem_index = []
        with open("./chem/index.pkl", "wb") as f:
            pickle.dump(chem_index, f)
    if not os.path.isdir("./bio"):
        os.mkdir("./bio")
    if os.path.isfile("./bio/index.pkl"):
        with open("./bio/index.pkl", "rb") as f:
            bio_index = pickle.load(f)
    else:
        bio_index = []
        with open("./bio/index.pkl", "wb") as f:
            pickle.dump(bio_index, f)
    index = {
        "chem": chem_index,
        "phy": phy_index,
        "math": math_index,
        "bio": bio_index
    }
    screen_size = options["size"]
    if options["online"]:
        client = AipOcr(options["APP_ID"], options["API_KEY"], options["SECRET_KEY"])
    else:
        client = ocr
    if options["erase"]:
        erase.set_key(options["yd_id"], options["yd_key"])
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    app.exec()
