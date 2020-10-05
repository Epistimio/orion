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
import React, { Component, Fragment } from 'react';
import PropTypes from 'prop-types';

import {
  ComposedModal,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
  InlineLoading
} from 'carbon-components-react';

class ActionModal extends Component {
  state = {};
  modalRef = React.createRef();

  onClickCapture = (evt) => {
    // Carbon's default is to close out of the modal when a click event happens outside.
    // They don't give us an easy way to override this with ComposedModal, so block the
    // click event altogether if it is outside the main modal container.
    if (evt && evt.target) {
      const innerModal = this.modalRef && this.modalRef.current && this.modalRef.current.innerModal;
      if (innerModal && innerModal.current && !innerModal.current.contains(evt.target)) {
        evt.stopPropagation();
      }
    }
  }

  render() {
    const {
      className, danger, secondaryButtonText, secondaryClassName,
      onRequestClose, onRequestSubmit, submitting, success
    } = this.props;

    return (
      <ComposedModal
        open={this.props.open}
        className={className}
        onKeyDown={(e) => /*check for esc key*/ e.which === 27 ? onRequestClose() : this.props.onKeyDown()}
        onClickCapture={this.onClickCapture}
        ref={this.modalRef}
        danger={danger}>
        <ModalHeader title={this.props.modalHeading} buttonOnClick={onRequestClose} />
        <ModalBody label={this.props.modalLabel}>
          {this.props.children}
        </ModalBody>
        <ModalFooter
          secondaryButtonText={(submitting || success) ? null : secondaryButtonText}
          secondaryClassName={`bx--btn--secondary${secondaryClassName ? (" " + secondaryClassName) : ""}`}
          onRequestClose={onRequestClose}
          onRequestSubmit={onRequestSubmit}>
          {/* If the action modal is used for form modals, you should make form fields readonly or disabled while submitting */}
          {(submitting || success) ? (
            <Fragment>
              <Button
                disabled
                className={(secondaryClassName || "")}
                kind="secondary">
                {secondaryButtonText}
              </Button>
              <InlineLoading
                style={{ marginLeft: "1rem", width: "50%" }}
                description={this.props.submittingText}
                status={success ? "active" : "finished"}
                onSuccess={this.props.onSuccess}
                successDelay={this.props.successDelay || 1000}
              />
            </Fragment>
          ) : (
              <Button
                data-modal-primary-focus
                onClick={onRequestSubmit}
                disabled={this.props.primaryButtonDisabled}
                kind={danger ? "danger" : "primary"}>
                {this.props.primaryButtonText}
              </Button>
            )}
        </ModalFooter>
      </ComposedModal>
    );
  }
}

ActionModal.propTypes = {
  // If this modal is open or not
  open: PropTypes.bool.isRequired,
  // Content to be placed in the ModalBody
  children: PropTypes.node,
  // Optional classname to be applied to the whole modal
  className: PropTypes.string,
  // Specify true if this modal will result in a resource being deleted
  danger: PropTypes.bool,
  // Optional function to handle key down events
  onKeyDown: PropTypes.func,
  // Title to be displayed on the modal
  modalHeading: PropTypes.string,
  // Specify an optional sub-label to be displayed in the ModalBody underneath the ModalHeader
  modalLabel: PropTypes.string,
  // Text displayed on the primary footer button
  primaryButtonText: PropTypes.string.isRequired,
  // Specify whether the primary button should be disabled
  primaryButtonDisabled: PropTypes.bool,
  // Text displayed on the secondary footer button
  secondaryButtonText: PropTypes.string,
  // Class name to give to the secondary footer button
  secondaryClassName: PropTypes.string,
  // Function for when the secondary/close button is clicked
  onRequestClose: PropTypes.func,
  // Function for when the primary submit button is clicked
  onRequestSubmit: PropTypes.func,
  // Used to display an inline-loading icon in the primary button while the modal is submitting (calling an API)
  submitting: PropTypes.bool,
  // Text to display in the primary button when the modal is being submitted
  submittingText: PropTypes.string,
  // Used to display a check icon once an API call is complete. It will display for one second before closing out the modal
  success: PropTypes.bool,
  // Amount of time (in milliseconds) until the modal is closed out after success becomes true
  successDelay: PropTypes.number,
  // Function when an API call is complete and the success icon delay is complete. Usually used to close out the modal
  onSuccess: PropTypes.func
}

ActionModal.defaultProps = {
  className: "",
  open: false,
  danger: false,
  onKeyDown: () => {},
  primaryButtonDisabled: false,
  onRequestClose: () => {},
  onRequestSubmit: () => {},
  submitting: false,
  submittingText: "Submitting...",
  success: false,
  onSuccess: () => {}
}

export default ActionModal;
