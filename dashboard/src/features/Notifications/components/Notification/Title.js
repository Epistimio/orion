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
import React from 'react';
import { injectIntl, FormattedMessage } from 'react-intl';
import { Link } from 'react-router-dom';
import { useSelector } from 'react-redux';

const eventResourceMap = {
  'training': 'models.combinedData',
  
};

// Show the title of the notification, or the name and a link to the resource if available.
// The title will default to the notification type (ex. "Success", "Error", etc) if the title isn't provided
const Title = React.memo(props => {
  const resource_str = props.resource_str || eventResourceMap[props.event_type];
  const [stateType, stateProp] = resource_str ? resource_str.split('.') : [null, null]; // first index will also be the pathname value
  const resourceList = useSelector(state => !props.resource_name && stateType && stateProp && state[stateType][stateProp]);

  if (props.resource_id && (props.resource_name || (resourceList && resourceList.byId[props.resource_id]))) {
    const { resource_id, task_id } = props;
    const resourceId = props.event_type === 'inferencing' ? task_id : resource_id;
    return (
      <h3 className="bx--toast-notification__title">
        <Link
          className="bx--link task__title-link"
          to={{
            pathname: `/${stateType}/${resource_id}`,
            state: { resourceId },
          }}>
          {props.resource_name || resourceList.byId[resource_id].name}
        </Link>
      </h3>
    );
  }
  const title = props.intl.messages[props.intl_str] ? props.intl_str : props.notice_type;
  return (
    <h3 className="bx--toast-notification__title">
      <FormattedMessage id={title} values={props.values} />
    </h3>
  );
});

export default injectIntl(Title);
