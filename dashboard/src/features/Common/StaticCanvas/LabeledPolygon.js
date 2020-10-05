/* IBM_SOURCE_PROLOG_BEGIN_TAG                                            */
/* *****************************************************************      */
/*                                                                        */
/* IBM Confidential                                                       */
/* OCO Source Materials                                                   */
/*                                                                        */
/* (C) Copyright IBM Corp. 2019, 2020                                     */
/*                                                                        */
/* The source code for this program is not published or otherwise         */
/* divested of its trade secrets, irrespective of what has been           */
/* deposited with the U.S. Copyright Office.                              */
/*                                                                        */
/* *****************************************************************      */
/* IBM_SOURCE_PROLOG_END_TAG                                              */
import { fabric } from 'fabric';

const defaultPolyOptions = {
  fontSize: 16,
  fontZoom: 1,
  stroke: '#000',
};

const defaultTextOptions = {
  fontSize: 16,
  fontWeight: 'bold',
  fontFamily: '"IBM Plex Sans", "Helvetica", "Arial", sans-serif',
  originX: 'left',
  originY: 'bottom',
  padding: 7,
};

class LabeledPolygon {
  constructor(canvas, polygonOptions = {}, textOptions = {}, miscOptions = {}) {
    this.canvas = canvas;

    if (miscOptions.zoom) {
      textOptions.fontSize = +((16 / miscOptions.zoom).toFixed(1));
      textOptions.padding = 7 / miscOptions.zoom;
    }

    polygonOptions.points.forEach((arr) => {
      canvas.add(
        this.polygon = new fabric.Polygon(arr, {
          ...defaultPolyOptions,
          fill: polygonOptions.fillColor.replace("x", "0.4"), // setting the opacity of the rgba color
          type: 'labeledPoly',
          parent: this,
        }),
        this.label = new fabric.Text(textOptions.label, {
          ...defaultTextOptions,
          ...textOptions,
          backgroundColor: textOptions.backgroundColor.replace("x", "0.8"), // setting the opacity of the rgba color
          fill: textOptions.fillColor,
          type: 'text',
          parent: this,
        }),
      );
    });
  }

  positionItems() {
    // method used to reposition the label after a canvas zoom
    this.label.originY = 'bottom';
    this.label.top = this.polygon.top + this.polygon.strokeWidth;
    // anchor the label at the highest point of the polygon
    this.polygon.points.forEach((point) => {
      if (this.polygon.top === point.y) {
        this.label.left = point.x;
      }
    });

    let labelCoords = this.label.calcCoords(true);
    if (labelCoords && labelCoords.tl) {
      const topY = this.label.top - (Math.abs(labelCoords.bl.y - labelCoords.tl.y));
      if (+topY.toFixed(0) <= 0) {
        // the label is going off the top part of the canvas, move it into the box instead
        this.label.originY = 'top';
        this.label.top = this.polygon.top;
      }
    }

    this.label.setCoords();
    this.label.bringToFront();
    this.canvas.renderAll();
  }
}

export default LabeledPolygon;
