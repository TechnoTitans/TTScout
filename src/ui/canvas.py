import tkinter as tk
import enum

from util import Color


class Canvas(tk.Canvas):
    class Position(enum.Enum):
        TOP_LEFT = enum.auto()
        TOP_RIGHT = enum.auto()
        BOTTOM_LEFT = enum.auto()
        BOTTOM_RIGHT = enum.auto()

    cursors = {
        Position.TOP_LEFT: 'size_nw_se',
        Position.TOP_RIGHT: 'size_ne_sw',
        Position.BOTTOM_LEFT: 'size_ne_sw',
        Position.BOTTOM_RIGHT: 'size_nw_se'
    }

    def __init__(self, *args, **kwargs):
        super(Canvas, self).__init__(*args, **kwargs)

        self.rect_color = Color.GREEN.hex
        self.config(bg=Color.ALMOST_BLACK.hex)

        self._tag = 'resize'
        # stores the resize points
        self.resizePoints = {}
        # current rect absolute coordinates (TL_x, TL_y, BR_x, BR_y)
        self.rect_absolute = (80, 50, 100, 100)
        # previous mouse coordinates
        self.previous = (0, 0)

        self.bind('<Motion>', self.update_cursor)
        self.bind('<1>', self.set_resize_point)
        self.bind('<ButtonRelease-1>', self.release)

        self._current_resize_rect = None
        self._current_point = None

        self.create_resize_rect()

    def set_rect_pos(self, rect: (int, int, int, int)):
        x0, y0, x1, y1 = rect
        cx0, cy0, cx1, cy1 = self.canvasx(x0), self.canvasy(y0), self.canvasx(x1), self.canvasy(y1)

        # TODO figure out why this weird offset is needed even though the passed in values are correct
        self.rect_absolute = (cx0 + 2, cy0 + 2, cx1 - 2, cy1 - 2)
        self._resize_internal(*self.rect_absolute)

    def create_resize_rect(self):  # adds a rect around the canvas item
        self._current_resize_rect = self.create_rectangle(
            *self.rect_absolute, tags=self._tag, outline=self.rect_color, width=3
        )

        bbox = self.bbox(self._current_resize_rect)

        # the below are the points at 4 corners of resize rect
        self.resizePoints[self.Position.TOP_LEFT] = self.create_oval(bbox[0] - 5, bbox[1] - 5, bbox[0] + 5,
                                                                     bbox[1] + 5,
                                                                     fill=self.rect_color, tags=self._tag)
        self.resizePoints[self.Position.TOP_RIGHT] = self.create_oval(bbox[2] - 5, bbox[1] - 5, bbox[2] + 5,
                                                                      bbox[1] + 5,
                                                                      fill=self.rect_color, tags=self._tag)
        self.resizePoints[self.Position.BOTTOM_RIGHT] = self.create_oval(bbox[2] - 5, bbox[3] - 5, bbox[2] + 5,
                                                                         bbox[3] + 5,
                                                                         fill=self.rect_color, tags=self._tag)
        self.resizePoints[self.Position.BOTTOM_LEFT] = self.create_oval(bbox[0] - 5, bbox[3] - 5, bbox[0] + 5,
                                                                        bbox[3] + 5,
                                                                        fill=self.rect_color, tags=self._tag)

    def update_cursor(self, event: tk.Event):
        # Update cursor when hovering over resize points
        point = self.check_in_points(event.x, event.y)
        if point:
            key = list(self.resizePoints.keys())[list(self.resizePoints.values()).index(point)]
            self.config(cursor=self.cursors[key])
        else:
            self.config(cursor='')

    def check_in_points(self, x, y):
        # check if the mouse is over the resizePoints
        for item in self.resizePoints.values():
            if self.check_in_bbox(item, x, y):
                return item

        return None

    def check_in_bbox(self, item, x, y):
        # check if (x, y) points are inside the bounding box
        box = self.bbox(item)
        return box[0] < x < box[2] and box[1] < y < box[3]

    def set_resize_point(self, event: tk.Event):
        self._current_point = self.check_in_points(event.x, event.y)

        if self._current_point is not None:
            self.bind('<B1-Motion>', self.resize)
        else:
            self.previous = (event.x, event.y)
            self.bind('<B1-Motion>', self.move_item)

    # noinspection PyUnusedLocal
    def release(self, event: tk.Event):
        self.tag_unbind(self._tag, '<B1-Motion>')
        self.unbind('<B1-Motion>')

    def _move_internal(self, x: int, y: int):
        self.move(self._current_resize_rect, x, y)
        self.update_resize_rect()

    def move_item(self, event: tk.Event):
        # move the canvas item
        xc, yc = self.canvasx(event.x), self.canvasy(event.y)

        self._move_internal(xc - self.previous[0], yc - self.previous[1])
        self.previous = (xc, yc)

    def update_resize_rect(self):
        # updates the position of the resize rectangle
        new_coord = self.bbox(self._current_resize_rect)
        self.rect_absolute = new_coord

        self.moveto(self.resizePoints[self.Position.TOP_LEFT], new_coord[0] - 5, new_coord[1] - 5)
        self.moveto(self.resizePoints[self.Position.TOP_RIGHT], new_coord[2] - 5, new_coord[1] - 5)
        self.moveto(self.resizePoints[self.Position.BOTTOM_RIGHT], new_coord[2] - 5, new_coord[3] - 5)
        self.moveto(self.resizePoints[self.Position.BOTTOM_LEFT], new_coord[0] - 5, new_coord[3] - 5)

    def _resize_internal(self, x0: float, y0: float, x1: float, y1: float):
        self.coords(self._current_resize_rect, x0, y0, x1, y1)
        self.update_resize_rect()

    def resize(self, event: tk.Event):  # resizes the canvas item
        item_coords = self.coords(self._current_resize_rect)

        if self.resizePoints[self.Position.TOP_LEFT] == self._current_point:
            self._resize_internal(event.x, event.y, item_coords[2], item_coords[3])
        elif self.resizePoints[self.Position.TOP_RIGHT] == self._current_point:
            self._resize_internal(item_coords[0], event.y, event.x, item_coords[3])
        elif self.resizePoints[self.Position.BOTTOM_RIGHT] == self._current_point:
            self._resize_internal(item_coords[0], item_coords[1], event.x, event.y)
        elif self.resizePoints[self.Position.BOTTOM_LEFT] == self._current_point:
            self._resize_internal(event.x, item_coords[1], item_coords[2], event.y)

        self.update_resize_rect()
