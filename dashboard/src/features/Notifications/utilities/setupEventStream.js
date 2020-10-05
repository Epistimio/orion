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
import { EventSourcePolyfill } from 'event-source-polyfill';
import { processEvent, processorMap } from './processSSEEvent';

let eventStream = null;

export const createEventStream = authToken => {
  if (!eventStream && authToken) {
    // in dev mode, make sure you go to the instance you're connecting to and accept the certificate if needed
    const eventsEndpoint = process && process.env && process.env.NODE_ENV === 'development' && process.env.VISION_SERVICE_API ? `${process.env.VISION_SERVICE_API}/api/events` : 'api/events';
    eventStream = new EventSourcePolyfill(eventsEndpoint, {
      'headers': {
        'X-Auth-Token': authToken,
      },
    });

    // To add a new event, add it to the processor map above
    eventStream.addEventListener('ping', () => {}); // this is used to keep the event listener alive

    Object.keys(processorMap).forEach(eventType => {
      eventStream.addEventListener(eventType, msg => processEvent(msg, eventType));
    });
  }
};

export const closeEventStream = () => {
  if (eventStream) {
    eventStream.close();
    eventStream = null;
  }
};
