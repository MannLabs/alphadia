import * as React from 'react'

import { useMethod } from '../logic/context'

import { useTheme } from "@mui/material";
import { useRef, useEffect } from 'react'

import { FixedSizeList } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import InfiniteLoader from 'react-window-infinite-loader';

import { Box, Typography } from '@mui/material'
import { useProfile } from '../logic/profile';

const WARNING_ANSI_CODE = '33;20';
const ERROR_ANSI_CODE = '31;20';

function replaceConfigFormatUnicodeEscapes(text) {
    // Replace Unicode escape sequences used in formatting the config with their actual characters
    return text
      .replace(/\\u2514/g, '└')  // Box drawing light up and right
      .replace(/\\u251c/g, '├')  // Box drawing light vertical and right
      .replace(/\\u2502/g, '│')  // Box drawing light vertical
      .replace(/\\u2500/g, '─')  // Box drawing light horizontal
  }

function parseConsoleOutput(input, theme) {
    const processedInput = replaceConfigFormatUnicodeEscapes(input);

    // Match ANSI escape sequences - either with closing code or until end of line
    const terminalColorsEscapeRegex = /\[(\d+;?\d*)m(.*?)(?:\[(\d+)m|$)/g;
    const matches = [...processedInput.matchAll(terminalColorsEscapeRegex)];

    if (matches.length === 0) {
      return processedInput;
    }
    let result = [];
    let currentIndex = 0;
    let elementCounter = 0;
    for (const match of matches) {
      const startIndex = match.index;
      const text = match[2];
      const spanStyle = {color: getColorCode(match[1], theme), fontFamily: "Roboto Mono"}
      // Push the text before the escape character pair
      if (startIndex > currentIndex) {
        const plainText = processedInput.substring(currentIndex, startIndex);
        result.push(
          <span key={`text-${elementCounter++}`} style={{fontFamily: "Roboto Mono"}}>{plainText}</span>
        );
      }
      // Push the colored span with the text
      result.push(
        <span key={`span-${elementCounter++}`} style={spanStyle}>
          {text}
        </span>
      );
      currentIndex = startIndex + match[0].length;
    }

    // Push any remaining plain text after the last escape character pair
    if (currentIndex < processedInput.length) {
      const remainingText = processedInput.substring(currentIndex);
      result.push(
        <span key={`text-${elementCounter++}`} style={{fontFamily: "Roboto Mono"}}>{remainingText}</span>
      );
    }

    return result;
  }

  function getColorCode(colorCode, theme) {
    // Map color codes to actual colors
    const colorMap = {
      '30;20': 'black',
      '31;20': theme.palette.mode === 'light' ? "rgb(200, 1, 0)" : 'rgb(250, 150, 136)' ,
      '32;20': theme.palette.mode === 'light' ?'rgb(76 211 26)' : 'rgb(168, 219, 114)',
      '33;20': theme.palette.mode === 'light' ?'rgb(253 167 0)' : 'rgb(254, 219, 119)',
      '34;20': 'blue',
      '35;20': 'magenta',
      '36;20': 'cyan',
      '37;20': 'white',
      '38;20': 'inherit',
    };

    return colorMap[colorCode] || 'inherit'; // Default color is black
  }

function applyCarriageReturns(items) {
    const result = [];
    for (const item of items) {
        if (result.length > 0 && result[result.length - 1].endsWith('\r')) {
            result[result.length - 1] = item;
        } else {
            result.push(item);
        }
    }
    return result;
}

const Output = () => {

    const [cmd, setCmd] = React.useState("")

    const method  = useMethod();
    const theme = useTheme();

    const [items, setItems] = React.useState([])
    const backendLengthRef = useRef(0);
    const [listRef, setListRef] = React.useState(null)
    const [scrollAttached, setScrollAttached] = React.useState(true)

    const profile = useProfile();

    const updateItems = (offset) => {
        window.electronAPI.getOutputRowsNew(-1,{limit:100, offset}).then((newItems) => {
            backendLengthRef.current += newItems.length;
            setItems(items => applyCarriageReturns([...items, ...newItems]));
        });
    }

    React.useEffect(() => {
        setItems([]);
        backendLengthRef.current = 0;

        let isMounted = true;

        const interval = setInterval(() => {
            window.electronAPI.getOutputLengthNew(-1).then((length) => {
                if (isMounted){
                    if (length > backendLengthRef.current) {
                        updateItems(backendLengthRef.current);
                    }
                    if (length < backendLengthRef.current) {
                        setItems([]);
                        backendLengthRef.current = 0;
                    }
                }
            });
        }, 100);

        return () => {
            isMounted = false;
            clearInterval(interval);
        }



        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);


    const onScroll = ({ scrollDirection, scrollOffset, scrollUpdateWasRequested }) => {
        // scrolling is attached to the end of the list
        // continue to keep it attached
        if (scrollUpdateWasRequested) {
            setScrollAttached(true);

        // scrolling was requested by the user
        } else {

            // The user is scrolling forward and the end of the list is near
            // attach scrolling to the end of the list
            if (scrollDirection === "forward") {

                // scrollOffset is the number of pixels from the top of the list
                // listRef?.props?.height is the height of the list in pixels
                // listRef?.props?.itemSize is the height of each item in pixels
                // bottomElement is the index of the bottom element in the list
                const bottomElement = (scrollOffset + listRef?.props?.height) / listRef?.props?.itemSize;

                // if the bottom element is within 5 elements of the end of the list
                if ((items.length - bottomElement) < 3)
                    setScrollAttached(true);

            // The user is scrolling somewhere in the middle of the list
            // detach scrolling from the end of the list
            } else {
                setScrollAttached(false);
            }
        }
    }

    // scroll to the end of the list when new items are added
    useEffect(() => {
        if (scrollAttached) {
            listRef?.scrollToItem?.(items.length, 'end')
        }

        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [listRef, items.length])

    const warningCount = React.useMemo(() => {
        return items.filter(line => line.includes(`[${WARNING_ANSI_CODE}m`)).length;
    }, [items]);

    const errorCount = React.useMemo(() => {
        return items.filter(line => line.includes(`[${ERROR_ANSI_CODE}m`)).length;
    }, [items]);

    const Row = ({ index, style }) => {
        const content = parseConsoleOutput((index >= items.length) ? "" : items[index], theme);
        return (
            <Box style={style} sx={{fontSize: "inherit", fontFamily: "Roboto Mono", lineHeight: "1.0rem", textWrap: "nowrap", overflow: "hidden"}}>
                {content}
            </Box>
        )
    };

    return (
    <Box sx={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        whiteSpace: "break-spaces",
        overflow: "hidden",
        fontFamily: "Roboto Mono",
        fontSize: "0.7rem",
        lineHeight: "1.0rem",
        }}>

        <Box sx={{ flex: 1, overflow: "hidden" }}>
            <InfiniteLoader
                isItemLoaded={(index) => (index < items.length)}
                loadMoreItems={()=>{}}
                itemCount={items.length}
                >
                {({ onItemsRendered, ref: setRef }) => (
            <AutoSizer>
                {({ height, width }) => (
                    <FixedSizeList
                        className="List"
                        height={height}
                        itemCount={items.length}
                        itemSize={15}
                        width={width}
                        onItemsRendered={onItemsRendered}
                        onScroll={onScroll}
                        ref={(list) => {

                        setRef?.(list);
                        setListRef(list);
                        }}
                    >
                        {Row}
                    </FixedSizeList>
                )}
            </AutoSizer>
                )}
            </InfiniteLoader>
        </Box>

        <Box sx={{
            display: "flex",
            justifyContent: "flex-end",
            alignItems: "center",
            px: 1,
            py: 0.5,
            borderTop: 1,
            borderColor: "divider",
            backgroundColor: "background.paper",
            gap: 2,
        }}>
            <Typography variant="caption" sx={{ fontFamily: "Roboto Mono", color: theme.palette.mode === 'light' ? 'rgb(253 167 0)' : 'rgb(254, 219, 119)' }}>
                Warnings: {warningCount}
            </Typography>
            <Typography variant="caption" sx={{ fontFamily: "Roboto Mono", color: theme.palette.mode === 'light' ? 'rgb(200, 1, 0)' : 'rgb(250, 150, 136)' }}>
                Errors: {errorCount}
            </Typography>
        </Box>
    </Box>
    )}

export default Output
