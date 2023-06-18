import sys
from PySide6.QtWidgets import QApplication
# from PySide6.QtGui import
from main_window import MainWindow
# import qdarkstyle
# from qt_material import apply_stylesheet


def main():
    app = QApplication(sys.argv)
    # app.setStyle('fusion')
    # dark_style = qdarkstyle.load_stylesheet(qt_api='PySide6')
    # app.setStyleSheet(dark_style)
    # apply_stylesheet(app, theme='dark_cyan.xml', invert_secondary=True, extra={'density_scale': '-2'})  #

    window = MainWindow()
    window.show()

    window.quit_app.connect(app.quit)

    app.exec()

    return 0


if __name__ == '__main__':
    # queries = [
    #     # 1
    #     "AND machine+learning, OR classification, OR categorization, OR clustering",
    #     # 2
    #     "AND recommender+systems, OR information+filtering, OR user+profile+construction, OR user+feedback",
    #     # 3
    #     "AND information+retrieval+systems, OR automated+retrieval+systems, OR information+overload,"
    #     " OR web+search+engine, OR question+answering",
    #     # 4
    #     "AND expert+systems, OR expert+estimates, OR expert+rules",
    #     # 5
    #     "AND neural+nets, OR artificial+neural+network",
    #     # 6
    #     "AND fuzzy+logic, OR fuzzy+sets, OR fuzzy+rules, OR membership+function, OR model+uncertainty,"
    #     " OR linguistic+variable",
    #     # 7
    #     "AND computational+complexity, OR data+structures+and+algorithms, OR computer+algorithms+analysis,"
    #     " OR efficient+algorithm",
    #     # 8
    #     "AND computer+vision, OR autonomous+robots"
    # ]
    main()
