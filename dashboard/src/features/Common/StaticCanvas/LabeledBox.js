/* IBM_SOURCE_PROLOG_BEGIN_TAG                                            */
/* *****************************************************************      */
/*                                                                        */
/* IBM Confidential                                                       */
/* OCO Source Materials                                                   */
/*                                                                        */
/* (C) Copyright IBM Corp. 2018, 2020                                     */
/*                                                                        */
/* The source code for this program is not published or otherwise         */
/* divested of its trade secrets, irrespective of what has been           */
/* deposited with the U.S. Copyright Office.                              */
/*                                                                        */
/* *****************************************************************      */
/* IBM_SOURCE_PROLOG_END_TAG                                              */
import { fabric } from 'fabric';

const strokeWidth = 2.5;

const defaultRectOptions = {
  fontSize: 16,
  fontZoom: 1,
  fill: 'transparent',
};

const defaultTextOptions = {
  fontSize: 16,
  fontWeight: 'bold',
  fontFamily: '"IBM Plex Sans", "Helvetica", "Arial", sans-serif',
  originX: 'left',
  originY: 'bottom',
  padding: 7,
};

class LabeledBox {
  constructor(canvas, rectOptions = {}, textOptions = {}, miscOptions = {}) {
    this.canvas = canvas;

    if (miscOptions.zoom) {
      textOptions.fontSize = +((16 / miscOptions.zoom).toFixed(1));
      textOptions.padding = 7 / miscOptions.zoom;
      rectOptions.strokeWidth = strokeWidth / miscOptions.zoom;
    }

    canvas.add(
      this.bndBox = new fabric.Rect({
        ...defaultRectOptions,
        ...rectOptions,
        stroke: rectOptions.stroke.replace('x', '1'), // setting the opacity of the rgba color
        type: 'labeledRect',
        parent: this,
        top: rectOptions.top,
        left: rectOptions.left,
        width: rectOptions.width,
        height: rectOptions.height,
      }),
      this.label = new fabric.Text(textOptions.label, {
        ...defaultTextOptions,
        ...textOptions,
        backgroundColor: textOptions.backgroundColor.replace('x', '0.8'),
        fill: textOptions.fillColor,
        type: 'text',
        parent: this,
      }),
    );
  }

  positionItems() {
    // method used to reposition the label after a canvas zoom
    const boxCoords = this.bndBox.calcCoords(true);

    this.label.left = boxCoords.tl.x;
    this.label.originY = 'bottom';
    this.label.top = boxCoords.tl.y + this.bndBox.strokeWidth;

    let labelCoords = this.label.calcCoords(true);
    if (labelCoords && labelCoords.tl) {
      const topY = this.label.top - (Math.abs(labelCoords.bl.y - labelCoords.tl.y));
      if (+topY.toFixed(0) <= 0) {
        // the label is going off the top part of the canvas, move it into the box instead
        this.label.originY = 'top';
        this.label.top = boxCoords.tl.y;
        this.label.left = boxCoords.tl.x + this.bndBox.strokeWidth;
      }
    }

    this.label.setCoords();
    this.label.bringToFront();
    this.canvas.renderAll();
  }
}

export default LabeledBox;
