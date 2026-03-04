#
# ABOUT
# Artisan Thermal Transfer Label PDF Generator
#
# Generates a small-format PDF suitable for printing on thermal transfer
# label printers (default 4" x 2"). Contains roast details such as batch
# number, bean name, date, weights, color, key events and notes.

# LICENSE
# This program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 2 of the License, or
# version 3 of the License, or (at your option) any later version. It is
# provided for educational purposes and is distributed in the hope that
# it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
# the GNU General Public License for more details.

import logging
from typing import Final, TYPE_CHECKING

if TYPE_CHECKING:
    from artisanlib.main import ApplicationWindow

from PyQt6.QtCore import Qt, QMarginsF, QRectF
from PyQt6.QtGui import QFont, QPageLayout, QPageSize, QPainter, QPen, QColor
from PyQt6.QtWidgets import QApplication

from artisanlib.util import stringfromseconds, float2float, float2str, weight_units, convertWeight

_log: Final[logging.Logger] = logging.getLogger(__name__)

# Label dimensions in points (1 point = 1/72 inch)
LABEL_WIDTH_IN = 4.0
LABEL_HEIGHT_IN = 2.0
MARGIN_IN = 0.10
DPI = 300


def _collect_roast_data(aw: 'ApplicationWindow') -> dict[str, str]:
    """Gather roast data from the application into a flat dict of display strings."""
    qmc = aw.qmc
    data: dict[str, str] = {}

    # Batch
    if qmc.roastbatchnr > 0:
        data['batch'] = f'{qmc.roastbatchprefix}{qmc.roastbatchnr}'
    else:
        data['batch'] = ''

    # Title / bean name
    data['title'] = qmc.title if qmc.title != QApplication.translate('Scope Title', 'Roaster Scope') else ''
    data['beans'] = qmc.beans

    # Date
    data['date'] = qmc.roastdate.toString('yyyy-MM-dd HH:mm')

    # Operator / Roaster / Organization
    data['operator'] = qmc.operator
    data['roaster'] = qmc.roastertype
    data['organization'] = qmc.organization

    # Weight in / out
    try:
        wunit = qmc.weight[2]
        w_in = qmc.weight[0]
        w_out = qmc.weight[1]
        if w_in > 0:
            data['weight_in'] = f'{float2float(w_in)}  {wunit}'
        else:
            data['weight_in'] = ''
        if w_out > 0:
            data['weight_out'] = f'{float2float(w_out)} {wunit}'
        else:
            data['weight_out'] = ''
        if w_in > 0 and w_out > 0:
            loss = aw.weight_loss(w_in, w_out)
            data['weight_loss'] = f'{float2float(loss, 1)}%'
        else:
            data['weight_loss'] = ''
    except Exception:  # pylint: disable=broad-except
        data['weight_in'] = ''
        data['weight_out'] = ''
        data['weight_loss'] = ''

    # Color
    color_parts: list[str] = []
    if qmc.whole_color:
        color_parts.append(f'W:{float2str(qmc.whole_color)}')
    if qmc.ground_color:
        color_parts.append(f'G:{float2str(qmc.ground_color)}')
    if color_parts and qmc.color_system_idx:
        color_parts.append(qmc.color_systems[qmc.color_system_idx])
    data['color'] = ' '.join(color_parts)

    # Density
    if qmc.density[0] != 0.0:
        data['density'] = f'{float2float(qmc.density[0])} g/l'
    else:
        data['density'] = ''

    # Moisture
    moisture_parts: list[str] = []
    if qmc.moisture_greens:
        moisture_parts.append(f'Green: {float2float(qmc.moisture_greens, 1)}%')
    if qmc.moisture_roasted:
        moisture_parts.append(f'Roasted: {float2float(qmc.moisture_roasted, 1)}%')
    data['moisture'] = '  '.join(moisture_parts)

    # Computed profile info for key times
    try:
        cp = aw.computedProfileInformation()
    except Exception:  # pylint: disable=broad-except
        cp = {}

    # Key roast events
    if 'FCs_time' in cp:
        data['fcs_time'] = stringfromseconds(cp['FCs_time'])
        if 'FCs_BT' in cp:
            data['fcs_temp'] = f"{cp['FCs_BT']:.0f}\u00b0{qmc.mode}"
        else:
            data['fcs_temp'] = ''
    else:
        data['fcs_time'] = ''
        data['fcs_temp'] = ''

    if 'DROP_time' in cp:
        data['drop_time'] = stringfromseconds(cp['DROP_time'])
        if 'DROP_BT' in cp:
            data['drop_temp'] = f"{cp['DROP_BT']:.0f}\u00b0{qmc.mode}"
        else:
            data['drop_temp'] = ''
    else:
        data['drop_time'] = ''
        data['drop_temp'] = ''

    if 'DRY_time' in cp:
        data['dry_time'] = stringfromseconds(cp['DRY_time'])
    else:
        data['dry_time'] = ''

    if 'totaltime' in cp:
        data['total_time'] = stringfromseconds(cp['totaltime'])
    else:
        data['total_time'] = ''

    # Development time ratio (time after FCs / total time)
    if 'FCs_time' in cp and 'DROP_time' in cp:
        dev_time = cp['DROP_time'] - cp['FCs_time']
        data['dev_time'] = stringfromseconds(dev_time)
        if cp['DROP_time'] > 0:
            data['dev_ratio'] = f'{(dev_time / cp["DROP_time"]) * 100:.0f}%'
        else:
            data['dev_ratio'] = ''
    else:
        data['dev_time'] = ''
        data['dev_ratio'] = ''

    # Roasting notes (truncated for label)
    notes = qmc.roastingnotes.strip()
    if len(notes) > 120:
        notes = notes[:117] + '...'
    data['notes'] = notes

    return data


def generate_label_pdf(aw: 'ApplicationWindow', filename: str) -> bool:
    """Generate a thermal transfer label PDF and save to *filename*.

    Returns True on success, False on failure.
    """
    try:
        from PyQt6.QtGui import QPdfWriter

        data = _collect_roast_data(aw)

        writer = QPdfWriter(filename)
        writer.setResolution(DPI)

        # Set custom page size for thermal label
        label_size = QPageSize(
            QPageSize.PageSizeId.Custom,
        )
        # Use explicit point size: 4in x 2in = 288pt x 144pt
        from PyQt6.QtCore import QSizeF
        page_size = QPageSize(QSizeF(LABEL_WIDTH_IN * 25.4, LABEL_HEIGHT_IN * 25.4), QPageSize.Unit.Millimeter, 'ThermalLabel')
        writer.setPageSize(page_size)

        margin_mm = MARGIN_IN * 25.4
        page_layout = QPageLayout(
            page_size,
            QPageLayout.Orientation.Landscape,
            QMarginsF(margin_mm, margin_mm, margin_mm, margin_mm),
            QPageLayout.Unit.Millimeter,
        )
        writer.setPageLayout(page_layout)

        painter = QPainter()
        if not painter.begin(writer):
            return False

        try:
            _draw_label(painter, writer, data)
        finally:
            painter.end()

        return True
    except Exception as e:  # pylint: disable=broad-except
        _log.exception(e)
        return False


def _draw_label(painter: QPainter, writer: 'QPdfWriter', data: dict[str, str]) -> None:
    """Draw the roast label content onto the painter."""
    from PyQt6.QtGui import QPdfWriter

    # Get the drawable area in device pixels
    page_rect = writer.pageLayout().paintRectPixels(writer.resolution())
    w = page_rect.width()
    h = page_rect.height()

    # Color constants
    black = QColor(0, 0, 0)
    dark_grey = QColor(80, 80, 80)
    light_grey = QColor(200, 200, 200)

    # Font setup
    def make_font(size_pt: float, bold: bool = False) -> QFont:
        f = QFont('Helvetica')
        f.setPointSizeF(size_pt)
        f.setBold(bold)
        return f

    font_title = make_font(14, bold=True)
    font_batch = make_font(11, bold=True)
    font_label = make_font(6.5, bold=True)
    font_value = make_font(7.5)
    font_notes = make_font(6)

    pen_black = QPen(black)
    pen_grey = QPen(dark_grey)
    pen_line = QPen(light_grey, 2)

    # Scale factor: points to device pixels
    sx = w / (LABEL_WIDTH_IN * 72 - 2 * MARGIN_IN * 72)
    sy = h / (LABEL_HEIGHT_IN * 72 - 2 * MARGIN_IN * 72)

    def px(pt: float) -> int:
        return int(pt * sx)

    def py(pt: float) -> int:
        return int(pt * sy)

    # --- HEADER ROW ---
    y = 0
    painter.setPen(pen_black)

    # Batch number (left)
    batch_text = data.get('batch', '')
    if batch_text:
        painter.setFont(font_batch)
        painter.drawText(QRectF(0, y, px(80), py(14)), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, batch_text)

    # Date (right)
    date_text = data.get('date', '')
    if date_text:
        painter.setFont(font_value)
        painter.drawText(QRectF(px(80), y, w - px(80), py(14)), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, date_text)

    y += py(14)

    # Title / Beans line
    display_title = data.get('beans', '') or data.get('title', '')
    if display_title:
        painter.setFont(font_title)
        painter.drawText(QRectF(0, y, w, py(18)), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, display_title)
    y += py(17)

    # Separator line
    painter.setPen(pen_line)
    painter.drawLine(0, y, w, y)
    y += py(3)

    # --- DATA GRID ---
    # We'll lay out data in two columns
    col1_x = 0
    col2_x = px(140)
    row_h = py(10)

    def draw_field(x: int, y_pos: int, label: str, value: str) -> None:
        if not value:
            return
        painter.setPen(pen_grey)
        painter.setFont(font_label)
        painter.drawText(QRectF(x, y_pos, px(45), row_h), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
        painter.setPen(pen_black)
        painter.setFont(font_value)
        painter.drawText(QRectF(x + px(45), y_pos, px(90), row_h), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, value)

    # Row 1: Weight In / FCs
    fcs_str = data.get('fcs_time', '')
    if fcs_str and data.get('fcs_temp', ''):
        fcs_str += f' @ {data["fcs_temp"]}'
    draw_field(col1_x, y, 'Weight In:', data.get('weight_in', ''))
    draw_field(col2_x, y, 'FCs:', fcs_str)
    y += row_h

    # Row 2: Weight Out / DROP
    drop_str = data.get('drop_time', '')
    if drop_str and data.get('drop_temp', ''):
        drop_str += f' @ {data["drop_temp"]}'
    draw_field(col1_x, y, 'Weight Out:', data.get('weight_out', ''))
    draw_field(col2_x, y, 'DROP:', drop_str)
    y += row_h

    # Row 3: Loss / Dev Time
    dev_str = data.get('dev_time', '')
    if dev_str and data.get('dev_ratio', ''):
        dev_str += f' ({data["dev_ratio"]})'
    draw_field(col1_x, y, 'Loss:', data.get('weight_loss', ''))
    draw_field(col2_x, y, 'Dev:', dev_str)
    y += row_h

    # Row 4: Color / Total Time
    draw_field(col1_x, y, 'Color:', data.get('color', ''))
    draw_field(col2_x, y, 'Total:', data.get('total_time', ''))
    y += row_h

    # Row 5: Density / Moisture
    draw_field(col1_x, y, 'Density:', data.get('density', ''))
    draw_field(col2_x, y, 'Moisture:', data.get('moisture', ''))
    y += row_h

    # Row 6: Roaster / Operator
    draw_field(col1_x, y, 'Roaster:', data.get('roaster', ''))
    draw_field(col2_x, y, 'Operator:', data.get('operator', ''))
    y += row_h

    # --- NOTES ---
    notes = data.get('notes', '')
    if notes:
        y += py(1)
        painter.setPen(pen_line)
        painter.drawLine(0, y, w, y)
        y += py(2)
        painter.setPen(pen_grey)
        painter.setFont(font_label)
        painter.drawText(QRectF(0, y, px(35), py(8)), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, 'Notes:')
        painter.setPen(pen_black)
        painter.setFont(font_notes)
        notes_rect = QRectF(px(35), y, w - px(35), h - y)
        painter.drawText(notes_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, notes)
