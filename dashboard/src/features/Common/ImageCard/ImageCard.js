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
import './styles.scss';

import React, { Component } from 'react';
import PropTypes from 'prop-types';
import classnames from 'classnames';

import { Checkbox } from 'carbon-components-react';

import emptyImage from 'images/image-file_128.svg';

class ImageCard extends Component {
  imgDiv = React.createRef();

  componentDidMount() {
    if (this.props.bgImgPath) {
      const image = new Image();

      image.src = this.props.bgImgPath;
      image.onload = () => {
        const imgContainer = this.imgDiv && this.imgDiv.current;
        if (imgContainer && image.width && image.height) {
          // valid image
          imgContainer.style.backgroundImage = `url('${this.props.bgImgPath}')`;
          imgContainer.classList.add('image-card__image');
        }
      }
    }
  }

  isCheckboxTarget(evt) {
    if (evt.target.type === "checkbox" || (evt.target.control && evt.target.control.type === "checkbox")) {
      // this onClick event was fired on the checkbox in the image card, ignore it
      return true;
    }
  }

  onCheckboxChange = (checked) => {
    // the onChange event should be passed in from the parent as an "onCheckedChange" prop
    if (this.props.onCheckedChange) {
      this.props.onCheckedChange(this.props.resourceId, checked);
    }
  }

  onCardClick = (evt) => {
    evt.persist();
    // the onClick event should be passed in from the parent as an "onCardClick" prop
    // don't handle checkbox events here since they're handled in the onChange event
    if (!this.isCheckboxTarget(evt)) {
      const checked = !this.props.checked;
      if (this.props.onTopContainerClick && this.topContainer.contains(evt.target)) {
        this.props.onTopContainerClick(this.props.resourceId, checked);
      } else if (this.props.onCardClick) {
        // at this point, they haven't clicked the title, secondary container, or checkbox.
        // send the checked value just in case it's needed
        this.props.onCardClick(this.props.resourceId, checked);
      }
    }
  }

  onKeyClick = (evt) => {
    evt.persist();
    // the onKeyDown event should be passed in from the parent as an "onKeyClick" prop
    // only call onKeyClick on a tile for the enter key
    if (this.props.onKeyClick && evt.key === "Enter" && !this.isCheckboxTarget(evt)) {
      // ensure the checkbox toggles when key clicking between the checkbox and the tile
      const checked = !this.props.checked;
      this.props.onKeyClick(this.props.resourceId, checked);
    }
  }

  render() {
    // if the image card respresents a video, we need to show different color style
    const tileClassNames = classnames({
      'image-card': true,
      'image-card--small': this.props.forDataTable,
      'image-card--selected': this.props.checked,
      'image-card--clickable': !!this.props.onCardClick
    });

    return (
      <div
        onClickCapture={this.onCardClick}
        onKeyDownCapture={this.onKeyClick}
        className={classnames(tileClassNames, this.props.className)}
        tabIndex="0">

        <div className="card-overlay__container">
          <div className="card-overlay__top-container" ref={node => this.topContainer = node}>
            <div className="card-overlay__main-container">
              <div className="card-overlay__checkbox--container">
                <Checkbox
                  onChange={this.onCheckboxChange}
                  checked={this.props.checked}
                  labelText=""
                  id={`${this.props.resourceId}_checkbox`}
                />
              </div>
              {!this.props.forDataTable && this.props.renderTitle("card-overlay__title")}
            </div>
            {!this.props.forDataTable && this.props.renderSecondary("card-overlay__secondary-container")}
          </div>
          {!this.props.forDataTable && this.props.renderTertiary("card-overlay__tertiary-container")}
        </div>

        {/* the empty images svg will only display if there is no dataset image or if it's still loading */}
        <svg className="image-card__no-image" viewBox={emptyImage.viewBox} fill="#E3E2E2" fillRule="evenodd">
          <use xlinkHref={`#${emptyImage.id}`}></use>
        </svg>
        {/* once the image is fetched, it will be set as a background-image on this div */}
        <div ref={this.imgDiv}></div>
      </div>
    )
  }
}

ImageCard.propTypes = {
  resourceId: PropTypes.string.isRequired,
  checked: PropTypes.bool.isRequired,
  forDataTable: PropTypes.bool,
  className: PropTypes.string,
  bgImgPath: PropTypes.string,
  renderTitle: PropTypes.func,
  renderSecondary: PropTypes.func,
  renderTertiary: PropTypes.func,
  onCardClick: PropTypes.func,
  onTopContainerClick: PropTypes.func,
  onKeyClick: PropTypes.func,
  onCheckedChange: PropTypes.func.isRequired,
};

ImageCard.defaultProps = {
  forDataTable: false,
  checked: false,
  renderTitle: () => { },
  renderSecondary: () => { },
  renderTertiary: () => { },
};

export default ImageCard;
