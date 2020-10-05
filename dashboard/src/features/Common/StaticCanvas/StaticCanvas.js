/* IBM_SOURCE_PROLOG_BEGIN_TAG                                            */
/* *****************************************************************      */
/*                                                                        */
/* IBM Confidential                                                       */
/* OCO Source Materials                                                   */
/*                                                                        */
/* (C) Copyright IBM Corp. 2018, 2019                                     */
/*                                                                        */
/* The source code for this program is not published or otherwise         */
/* divested of its trade secrets, irrespective of what has been           */
/* deposited with the U.S. Copyright Office.                              */
/*                                                                        */
/* *****************************************************************      */
/* IBM_SOURCE_PROLOG_END_TAG                                              */
import React, { Component, Fragment } from 'react';
import { fabric } from 'fabric';

import { Loading } from 'carbon-components-react';

import LabeledBox from './LabeledBox';
import LabeledPolygon from './LabeledPolygon';

// Fabric text and textboxes don't use the 'padding' property correctly without this override
fabric.Text.prototype.set({
  _getNonTransformedDimensions() {
    return new fabric.Point(this.width, this.height).scalarAdd(this.padding);
  },
  _calculateCurrentDimensions() {
    return fabric.util.transformPoint(this._getTransformedDimensions(), this.getViewportTransform(), true);
  }
});

class StaticCanvas extends Component {
  state = {
    canvas: null,
    firstUpload: true,
    loadedImg: false,
    zoomPct: 1,
  };

  componentDidMount() {
    let canvas = new fabric.StaticCanvas("static-canvas", {
      strokeWidth: 2.5
    });

    this.setState({
      canvas,
    });
  }

  componentDidUpdate(prevProps, prevState) {
    if ((!prevState.canvas && this.state.canvas)
      || (!prevProps.imgLabels && this.props.imgLabels)
      || (!prevProps.maxWidth && this.props.maxWidth)) {
      // we just got some new information...check if we can set the canvas image yet
      if (this.state.canvas && this.props.maxWidth && this.props.imgLabels && this.state.firstUpload) {
        // first upload of the image onto the canvas
        this.setCanvasImage();
        this.setState({
          firstUpload: false,
        });
      } else {
        // the image is the same, but the labels have updated
        this.updateBoxes();
      }
    } else if ((prevProps.maxHeight || prevProps.maxWidth) && (prevProps.maxHeight !== this.props.maxHeight || prevProps.maxWidth !== this.props.maxWidth)) {
      // the window size was updated, clear out the canvas and reset the image/labels
      this.resetCanvas();
      this.setCanvasImage();
    } else if (prevProps.imgPath && prevProps.imgPath !== this.props.imgPath) {
      // the image has been updated; clear out the canvas and wait for the new data to come in
      this.resetCanvas();
      this.setState({
        loadedImg: false,
        firstUpload: true,
      });
    } else if (this.props.imgLabels && prevProps.imgLabels && prevProps.imgLabels.length !== this.props.imgLabels.length) {
      this.updateBoxes(); // used when showing auto labels that were originally filtered out
    }
  }

  componentWillUnmount() {
    if (this.state.canvas) {
      // destroy canvas and remove all event listeners
      this.state.canvas.dispose();
    }
  }

  resetCanvas() {
    this.state.canvas.clear();
    this.state.canvas.viewportTransform = [1, 0, 0, 1, 0, 0];
    this.state.canvas.strokeWidth = 2.5;
  }

  setCanvasImage() {
    const { canvas } = this.state;
    const { imgHeight, imgWidth, imgPath, imgId } = this.props;

    this.setState({
      loadedImg: false
    });

    fabric.Image.fromURL(imgPath, (imgInstance) => {
      if (this.props.imgId !== imgId || !canvas) {
        // they selected a different image or closed out of the page
        // while this one was loading, don't do anything
        return;
      }

      let zoomPctH = 1;
      let zoomPctW = 1;
      if (this.props.maxHeight && +imgHeight > this.props.maxHeight) {
        zoomPctH = this.props.maxHeight / imgHeight;
      } else if (this.props.minHeight && (+imgHeight < this.props.minHeight)) {
        zoomPctH = this.props.minHeight / imgHeight;
      }

      if (this.props.maxWidth && +imgWidth > this.props.maxWidth) {
        zoomPctW = this.props.maxWidth / imgWidth;
      } else if (this.props.minWidth && (+imgWidth < this.props.minWidth)) {
        zoomPctW = this.props.minWidth / imgWidth;
      }

      let zoomPct = Math.min(zoomPctH, zoomPctW);
      canvas.setHeight(imgHeight * zoomPct);
      canvas.setWidth(imgWidth * zoomPct);

      canvas.setBackgroundImage(imgInstance);
      let labelType = this.createLabels(zoomPct);

      // once all the objects have been created, set the zoom so that the image
      // fits within the canvas height or width by default
      canvas.setZoom(zoomPct);
      this.updateTextSize(zoomPct, labelType);

      let container = this.props.getContainer();
      container.style.height = `${canvas.height}px`;
      container.style.width = `${canvas.width}px`;
      canvas.renderAll();

      this.setState({
        zoomPct: zoomPct,
        loadedImg: true,
      });
    })
  }

  createLabels(zoomPct) {
    // default label type is bounding box
    let labelType = [];
    this.props.imgLabels && this.props.imgLabels.forEach(label => {
      const labelColor = label.color ? label.color : ((label.generate_type === "auto") ? "rgba(253, 130, 133, x)" : "rgba(60, 109, 240, x)");
      const textColor = this.getTextColor(labelColor);

      if (label.segment_polygons) {
        // add polygon to list of used labels in the static canvas
        if (labelType.indexOf("labeledPoly") === -1) {
          labelType.push("labeledPoly");
        }
        // create polygons with a label if available
        new LabeledPolygon(
          this.state.canvas,
          { points: label.segment_polygons, fillColor: labelColor, stroke: "#fff" },
          { label: label.confidence ? `${label.name} (${label.confidence})` : label.name, backgroundColor: labelColor, fillColor: textColor },
          { zoom: zoomPct }
        );
      } else {
        // add rectangle to list of used labels in the static canvas
        if (labelType.indexOf("labeledRect") === -1) {
          labelType.push("labeledRect");
        }
        // create a box with a label based on the information for the img
        const xmin = parseFloat(label.bndbox.xmin);
        const ymin = parseFloat(label.bndbox.ymin);
        const xmax = parseFloat(label.bndbox.xmax);
        const ymax = parseFloat(label.bndbox.ymax);

        // make sure we have the right plot points before creating the box
        if (!isNaN(parseInt(xmin)) && !isNaN(parseInt(ymin)) && !isNaN(parseInt(xmax)) && !isNaN(parseInt(ymax))) {
          new LabeledBox(
            this.state.canvas,
            { left: xmin, top: ymin, width: (xmax - xmin), height: (ymax - ymin), stroke: labelColor },
            { label: label.confidence ? `${label.name} (${label.confidence})` : label.name, backgroundColor: labelColor, fillColor: textColor },
            { zoom: zoomPct }
          );
        }
      }
    });
    return labelType;
  }

  getTextColor(backgroundColor) {
    let brightness = 0;
    // parse through rgba string to grab the numeric values
    let rgb = backgroundColor.match(/rgba?\((\d{1,3}), ?(\d{1,3}), ?(\d{1,3})\)?(?:, ?(\d(?:\.\d?))\))?/);
    if (rgb) {
      brightness = Math.round(((parseInt(rgb[1]) * 299) + (parseInt(rgb[2]) * 587) + (parseInt(rgb[3]) * 114)) / 1000);
    }
    return ((brightness > 125) ? '#000' : '#FFF');
  }

  updateTextSize(newZoom, labelType) {
    // the font size can't always be a set value because it will appear
    // very large when zooming in and very small when zooming out.
    // instead, we have to make the font size relative to the zoom value
    // so that it appears to always be the same size to the user.
    const { canvas } = this.state;
    const strokeWidth = (labelType.includes("labeledPoly")) ? 1.5 : +((2.5 / newZoom).toFixed(1));

    canvas.strokeWidth = strokeWidth;
    labelType.forEach((label) => {
      canvas.getObjects(label).forEach(obj => {
        obj.strokeWidth = strokeWidth;
      });
    });
    canvas.getObjects("text").forEach(obj => {
      const baseFontSize = this.props.fontSize || 16;
      obj.fontSize = +((baseFontSize / newZoom).toFixed(1)) || 1;
      obj.padding = +((7 / newZoom).toFixed(1)) || 1;
      obj.scaleX = 1;
      obj.scaleY = 1;
      obj._clearCache();
    });
    labelType.forEach((label) => {
      canvas.getObjects(label).forEach(obj => {
        // once all the new values are set, go back and re-position the label text
        obj.parent.positionItems();
      });
    });
    canvas.renderAll();
  }

  updateBoxes() {
    const { canvas, zoomPct } = this.state;
    // remove previous boxes (or polygons) and there corresponding text labels
    canvas.remove(...canvas.getObjects());
    // add the new labels to the canvas
    let labelType = this.createLabels(zoomPct);
    this.updateTextSize(zoomPct, labelType);
    canvas.renderAll();
  }

  render() {
    return (
      <Fragment>
        {!this.state.loadedImg &&
          <Loading withOverlay={false} />
        }
        <canvas id="static-canvas" />
      </Fragment>
    );
  }
}

export default StaticCanvas;
